from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    GlobalAttention,
    Set2Set,
)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        num_layers: int = 2,
        use_batch_norm: bool = False,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        act = _get_activation(activation)
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[idx + 1]))
                layers.append(act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeFNNConv(MessagePassing):
    """Implements: f^{t+1}(v) = FNN( sigma( f^t(v) W1 + sum_w FNN(f^t(v), f^t(w), e_vw) ) )."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "silu",
    ) -> None:
        super().__init__(aggr="add")
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.message_mlp = MLP(
            in_dim * 2 + edge_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            num_layers=2,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )
        self.update_mlp = MLP(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            num_layers=2,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )
        self.act = _get_activation(activation)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x_proj = self.node_proj(x)
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.act(x_proj + agg)
        return self.update_mlp(out)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)


class MultiTowerConv(MessagePassing):
    """Multiple tower message passing (Gilmer et al., 2017, Section 5.4).

    Splits node embeddings into *k* towers of dimension d/k.  Each tower has
    its own message MLP, node projection, and update MLP, but all towers share
    a single ``propagate()`` call so the expensive scatter/gather kernel is
    dispatched only once regardless of *k*.  Tower outputs are concatenated and
    mixed through a shared MLP *g*.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_towers: int = 8,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "silu",
    ) -> None:
        super().__init__(aggr="add")
        if hidden_dim % num_towers != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_towers ({num_towers})"
            )

        self.num_towers = num_towers
        self.tower_dim = hidden_dim // num_towers
        td = self.tower_dim

        self.node_projs = nn.ModuleList(
            nn.Linear(td, td) for _ in range(num_towers)
        )
        self.message_mlps = nn.ModuleList(
            MLP(
                td * 2 + edge_dim, td, td,
                dropout=dropout, num_layers=2,
                use_batch_norm=use_batch_norm, activation=activation,
            )
            for _ in range(num_towers)
        )
        self.update_mlps = nn.ModuleList(
            MLP(
                td, td, td,
                dropout=dropout, num_layers=2,
                use_batch_norm=use_batch_norm, activation=activation,
            )
            for _ in range(num_towers)
        )
        self.act = _get_activation(activation)

        self.mix = MLP(
            hidden_dim, hidden_dim, hidden_dim,
            dropout=dropout, num_layers=2,
            use_batch_norm=use_batch_norm, activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        chunks = x.chunk(self.num_towers, dim=-1)
        x_proj = torch.cat(
            [self.node_projs[i](chunks[i]) for i in range(self.num_towers)],
            dim=-1,
        )

        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        out = self.act(x_proj + agg)

        out_chunks = out.chunk(self.num_towers, dim=-1)
        updated = torch.cat(
            [self.update_mlps[i](out_chunks[i]) for i in range(self.num_towers)],
            dim=-1,
        )
        return self.mix(updated)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        xi_chunks = x_i.chunk(self.num_towers, dim=-1)
        xj_chunks = x_j.chunk(self.num_towers, dim=-1)
        msgs = [
            self.message_mlps[i](
                torch.cat([xi_chunks[i], xj_chunks[i], edge_attr], dim=-1)
            )
            for i in range(self.num_towers)
        ]
        return torch.cat(msgs, dim=-1)


class MultiTowerGNNNodeRegressor(nn.Module):
    """Node-level regressor using multiple-tower message passing."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_towers: int = 8,
        out_dim: int = 1,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_mlp = MLP(
            in_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )

        self.convs = nn.ModuleList(
            MultiTowerConv(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                num_towers=num_towers,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                activation=activation,
            )
            for _ in range(num_layers)
        )

        self.output_mlp = MLP(
            hidden_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            num_layers=2,
            activation=activation,
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.input_mlp(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        return self.output_mlp(x)


def bidirectional_edge_pairs(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """For each edge (i, j), append the reverse (j, i) with the same edge attributes.

    Use this when graphs were stored with one orientation per undirected pair but
    message passing should see both directions (standard PyG undirected practice).
    """
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    rev = edge_index.flip(0)
    out_index = torch.cat([edge_index, rev], dim=1)
    if edge_attr is None:
        return out_index, None
    out_attr = torch.cat([edge_attr, edge_attr], dim=0)
    return out_index, out_attr


class BidirectionalGNNNodeRegressor(nn.Module):
    """Same architecture as :class:`GNNNodeRegressor`; doubles each edge in the forward pass."""

    def __init__(self, model: "GNNNodeRegressor") -> None:
        super().__init__()
        self.model = model

    def forward(self, data) -> torch.Tensor:
        ei, ea = bidirectional_edge_pairs(data.edge_index, data.edge_attr)
        saved_i, saved_a = data.edge_index, data.edge_attr
        data.edge_index, data.edge_attr = ei, ea
        try:
            return self.model(data)
        finally:
            data.edge_index, data.edge_attr = saved_i, saved_a


def _build_pool(
    aggregation: str,
    hidden_dim: int,
    dropout: float,
    use_batch_norm: bool,
    set2set_steps: int,
):
    aggregation = aggregation.lower()
    if aggregation == "mean":
        return global_mean_pool, hidden_dim
    if aggregation in {"sum", "add"}:
        return global_add_pool, hidden_dim
    if aggregation == "max":
        return global_max_pool, hidden_dim
    if aggregation == "attention":
        gate_nn = MLP(
            hidden_dim,
            hidden_dim,
            1,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        return GlobalAttention(gate_nn=gate_nn), hidden_dim
    if aggregation == "set2set":
        return Set2Set(hidden_dim, processing_steps=set2set_steps), 2 * hidden_dim
    raise ValueError("Unsupported aggregation. Choose from: mean, sum, max, attention, set2set.")


class GNNGraphRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        out_dim: int = 1,
        aggregation: str = "mean",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        set2set_steps: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_mlp = MLP(
            in_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EdgeFNNConv(
                    hidden_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    activation=activation,
                )
            )

        self.pool, pool_out_dim = _build_pool(
            aggregation=aggregation,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            set2set_steps=set2set_steps,
        )

        self.output_mlp = MLP(
            pool_out_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            num_layers=2,
            activation=activation,
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = self.input_mlp(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        if isinstance(self.pool, (GlobalAttention, Set2Set)):
            graph_repr = self.pool(x, batch)
        else:
            graph_repr = self.pool(x, batch)

        return self.output_mlp(graph_repr)


class BidirectionalGNNGraphRegressor(nn.Module):
    """Same architecture as :class:`GNNGraphRegressor`; doubles each edge in the forward pass."""

    def __init__(self, model: GNNGraphRegressor) -> None:
        super().__init__()
        self.model = model

    def forward(self, data) -> torch.Tensor:
        ei, ea = bidirectional_edge_pairs(data.edge_index, data.edge_attr)
        saved_i, saved_a = data.edge_index, data.edge_attr
        data.edge_index, data.edge_attr = ei, ea
        try:
            return self.model(data)
        finally:
            data.edge_index, data.edge_attr = saved_i, saved_a


class GNNNodeRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        out_dim: int = 1,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_mlp = MLP(
            in_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EdgeFNNConv(
                    hidden_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    activation=activation,
                )
            )

        self.output_mlp = MLP(
            hidden_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            num_layers=2,
            activation=activation,
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.input_mlp(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        return self.output_mlp(x)


def build_model_from_dataset(
    dataset,
    hidden_dim: int = 128,
    num_layers: int = 3,
    out_dim: int = 1,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    activation: str = "silu",
    bidirectional: bool = False,
) -> Union[GNNNodeRegressor, BidirectionalGNNNodeRegressor]:
    sample = dataset[0]
    in_dim = sample.x.size(-1)
    edge_dim = sample.edge_attr.size(-1)
    inner = GNNNodeRegressor(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=out_dim,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation,
    )
    if bidirectional:
        return BidirectionalGNNNodeRegressor(inner)
    return inner


def build_tower_model_from_dataset(
    dataset,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_towers: int = 8,
    out_dim: int = 1,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    activation: str = "silu",
    bidirectional: bool = False,
) -> nn.Module:
    """Build a :class:`MultiTowerGNNNodeRegressor`, optionally bidirectional."""
    sample = dataset[0]
    in_dim = sample.x.size(-1)
    edge_dim = sample.edge_attr.size(-1)
    inner = MultiTowerGNNNodeRegressor(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_towers=num_towers,
        out_dim=out_dim,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation,
    )
    if bidirectional:
        return BidirectionalGNNNodeRegressor(inner)
    return inner


class GatedConv(MessagePassing):
    """Gated graph convolution from Xie & Grossman (2018), CGCNN Eq. 5.

    z_{(i,j)} = v_i || v_j || u_{(i,j)}
    v_i^{t+1} = v_i^t + sum_j  sigma(z W_f + b_f) * g(z W_s + b_s)

    The sigmoid gate learns per-neighbor importance weights while the
    residual connection preserves gradient flow through deeper stacks.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "silu",
    ) -> None:
        super().__init__(aggr="add")
        z_dim = hidden_dim * 2 + edge_dim
        self.lin_f = nn.Linear(z_dim, hidden_dim)
        self.lin_s = nn.Linear(z_dim, hidden_dim)
        self.act = _get_activation(activation)

        self.bn = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = x + out
        if self.bn is not None:
            out = self.bn(out)
        if self.drop is not None:
            out = self.drop(out)
        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        gate = torch.sigmoid(self.lin_f(z))
        content = self.act(self.lin_s(z))
        return gate * content


class GatedGNNNodeRegressor(nn.Module):
    """Node-level regressor using CGCNN-style gated convolutions."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        out_dim: int = 1,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_mlp = MLP(
            in_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )

        self.convs = nn.ModuleList(
            GatedConv(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                activation=activation,
            )
            for _ in range(num_layers)
        )

        self.output_mlp = MLP(
            hidden_dim,
            hidden_dim,
            out_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            num_layers=2,
            activation=activation,
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.input_mlp(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        return self.output_mlp(x)


def build_gated_model_from_dataset(
    dataset,
    hidden_dim: int = 128,
    num_layers: int = 2,
    out_dim: int = 1,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    activation: str = "silu",
    bidirectional: bool = False,
) -> nn.Module:
    """Build a :class:`GatedGNNNodeRegressor`, optionally bidirectional."""
    sample = dataset[0]
    in_dim = sample.x.size(-1)
    edge_dim = sample.edge_attr.size(-1)
    inner = GatedGNNNodeRegressor(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=out_dim,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation,
    )
    if bidirectional:
        return BidirectionalGNNNodeRegressor(inner)
    return inner


class UAEGNNWrapper(nn.Module):
    """Replace the raw LAMMPS-type scalar with a ct-UAE-style atom embedding.

    The wrapper expects each PyG ``Data`` object to carry a ``z`` long tensor
    (set by :func:`graph_maker.build_pyg_dataset`; ``0`` = vacancy,
    otherwise the atomic number).  It calls
    :class:`ct_uae_pretrain.UAEAtomEncoder` on ``data.z`` and substitutes
    the result for the first column of ``data.x`` (if
    ``drop_type_scalar=True``, the default) or concatenates it in front
    of the full feature vector (otherwise).  The inner regressor sees a
    fresh ``data.x`` and is called unchanged.
    """

    def __init__(
        self,
        inner: nn.Module,
        uae_encoder: nn.Module,
        drop_type_scalar: bool = True,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.uae = uae_encoder
        self.drop_type_scalar = drop_type_scalar

    def forward(self, data) -> torch.Tensor:
        if not hasattr(data, "z") or data.z is None:
            raise RuntimeError(
                "UAEGNNWrapper requires Data.z (per-node atomic number). "
                "Rebuild the dataset with the current graph_maker.py."
            )
        z_emb = self.uae(data.z)
        if self.drop_type_scalar:
            rest = data.x[:, 1:]
        else:
            rest = data.x
        new_x = torch.cat([z_emb, rest], dim=-1)
        saved_x = data.x
        data.x = new_x
        try:
            return self.inner(data)
        finally:
            data.x = saved_x


def _uae_feature_dim(
    dataset,
    uae_emb_dim: int,
    drop_type_scalar: bool,
) -> int:
    base = dataset[0].x.size(-1)
    return uae_emb_dim + (base - 1 if drop_type_scalar else base)


def build_uae_gated_model_from_dataset(
    dataset,
    uae_ckpt_path: Optional[str] = None,
    uae_emb_dim: int = 8,
    uae_vocab_size: int = 100,
    freeze_uae: bool = True,
    drop_type_scalar: bool = True,
    hidden_dim: int = 128,
    num_layers: int = 2,
    out_dim: int = 1,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    activation: str = "silu",
    bidirectional: bool = True,
) -> nn.Module:
    """Build a :class:`GatedGNNNodeRegressor` fronted by a UAE encoder.

    Defaults correspond to the winning configuration found empirically on
    the binary-alloy PE dataset: 8-D frozen pretrained UAE, type scalar
    dropped, bidirectional gated CGCNN with ``hidden_dim=128`` and two
    layers.  Other widths / trainable setups were measured to underperform
    (see ``uae_gated_result.json`` for the reference run).
    """
    from ct_uae_pretrain import UAEAtomEncoder  # deferred to avoid circularity

    sample = dataset[0]
    if not hasattr(sample, "z") or sample.z is None:
        raise RuntimeError(
            "Dataset is missing per-node atomic numbers (Data.z). "
            "Rebuild with the updated adv_graph_maker.py."
        )
    in_dim = _uae_feature_dim(dataset, uae_emb_dim, drop_type_scalar)
    edge_dim = sample.edge_attr.size(-1)
    inner = GatedGNNNodeRegressor(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=out_dim,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation,
    )
    encoder = UAEAtomEncoder(
        ckpt_path=uae_ckpt_path,
        vocab_size=uae_vocab_size,
        emb_dim=uae_emb_dim,
        freeze=freeze_uae,
    )
    wrapped: nn.Module = UAEGNNWrapper(inner, encoder, drop_type_scalar)
    if bidirectional:
        wrapped = BidirectionalGNNNodeRegressor(wrapped)
    return wrapped


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name in {"leaky_relu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(
        "Unsupported activation. Choose from: relu, silu, gelu, tanh, elu, leaky_relu."
    )

