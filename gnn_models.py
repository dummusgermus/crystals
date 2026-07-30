from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    GlobalAttention,
    Set2Set,
)
from torch_geometric.utils import scatter, to_dense_adj, to_dense_batch


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


def _normalize_gts_activation(name: str) -> str:
    name = name.lower()
    if name in {"relu", "gelu"}:
        return name
    return "relu"


# ---------------------------------------------------------------------------
# Graph Transformer (TransformerEncoder + edge attention bias)
# Scalable adaptation of the pe_transformers / GDT-style node regressor.
# ---------------------------------------------------------------------------


class GraphTransformerNodeRegressor(nn.Module):
    """Node-level PE regressor using a standard TransformerEncoder.

    Nodes are packed into dense batches; edge attributes are projected to a
    per-head additive attention bias (same idea as the PE / GDT wrapper in
    ``tests/train_base_best_vs_pe_json.py``, without positional encodings or
    CLS tokens). Suitable for CPU/GPU training on our defect graphs.
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        out_dim: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_heads = num_heads
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, num_heads)
        act = activation.lower()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation=act,
            batch_first=True,
            norm_first=True,
        )
        # Disable nested-tensor path; we always pass an additive attn mask.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.attention_dropout = attention_dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU() if act == "gelu" else _get_activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data) -> torch.Tensor:
        if getattr(data, "batch", None) is None:
            data.batch = data.x.new_zeros(data.x.size(0), dtype=torch.long)

        x = self.node_encoder(data.x)
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = torch.ones(
                (data.edge_index.size(1), 1),
                dtype=data.x.dtype,
                device=data.x.device,
            )
        edge_bias = self.edge_encoder(edge_attr)

        x_pad, node_mask = to_dense_batch(x, data.batch)  # (B, N, H), (B, N)
        # (B, N, N, heads) additive attention bias from edges.
        attn = to_dense_adj(data.edge_index, data.batch, edge_bias)

        # Mask padded keys/queries with -inf (PE/GDT style). Do NOT also pass
        # src_key_padding_mask — mixing bool padding mask with float attn mask
        # yields NaNs on padded query rows.
        pad = ~node_mask
        attn = attn.masked_fill(pad.unsqueeze(2).unsqueeze(-1), float("-inf"))
        attn = attn.masked_fill(pad.unsqueeze(1).unsqueeze(-1), float("-inf"))
        # Keep self-attention on pad positions finite so softmax is defined;
        # their outputs are discarded via node_mask below.
        eye = torch.eye(attn.size(1), device=attn.device, dtype=torch.bool)
        attn = attn.masked_fill(eye.view(1, attn.size(1), attn.size(1), 1), 0.0)

        attn_mask = attn.permute(0, 3, 1, 2).reshape(
            -1, attn.size(1), attn.size(2)
        )

        out = self.encoder(x_pad, mask=attn_mask)
        if self.attention_dropout > 0:
            out = F.dropout(out, p=self.attention_dropout, training=self.training)
        return self.head(out[node_mask])


def build_graph_transformer_from_dataset(
    dataset,
    hidden_dim: int = 128,
    num_layers: int = 4,
    out_dim: int = 1,
    num_heads: int = 4,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
    activation: str = "gelu",
) -> GraphTransformerNodeRegressor:
    sample = dataset[0]
    in_dim = sample.x.size(-1)
    edge_dim = sample.edge_attr.size(-1) if sample.edge_attr is not None else 1
    return GraphTransformerNodeRegressor(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=out_dim,
        num_heads=num_heads,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation=activation,
    )


# ---------------------------------------------------------------------------
# GTS triangular-attention transformer (from tests/gnn_models_hadamard.py)
# Dense O(N^3) attention — prefer GPU; use GraphTransformer on CPU.
# ---------------------------------------------------------------------------


class _GTSMLP(nn.Sequential):
    def __init__(
        self, input_dim: int, output_dim: int, dropout: float = 0.0, linear: bool = False
    ):
        if linear:
            super().__init__(nn.Linear(input_dim, output_dim))
            return
        hidden_dim = output_dim
        super().__init__(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )


class _GTSFeatureEncoder(nn.Module):
    def __init__(self, embed_dim: int, node_dim: int, edge_dim: int):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, embed_dim)
        self.edge_encoder = nn.Linear(edge_dim, embed_dim)

    def forward(self, data):
        data.x = self.node_encoder(data.x)
        if not hasattr(data, "edge_attr") or data.edge_attr is None:
            data.edge_attr = torch.ones(
                (data.edge_index.size(1), 1), device=data.x.device
            )
        data.edge_attr = self.edge_encoder(data.edge_attr)
        return data


class _GTSComposer(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.node_proj = _GTSMLP(2 * embed_dim, embed_dim, linear=True)

    def forward(self, x, edge_index, edge_attr, batch, token_index, token_attr=None):
        edge_features = to_dense_adj(edge_index, batch, edge_attr)
        if token_attr is not None:
            token_attr = to_dense_adj(token_index, batch, token_attr)
            edge_features = torch.cat([edge_features, token_attr], -1)
        x = x[token_index.T].flatten(1, 2)
        x = self.node_proj(x)
        x = to_dense_adj(token_index, batch, x)
        return x + edge_features


class _GTSFFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
        norm: str = "layer",
    ):
        super().__init__()
        activation = _normalize_gts_activation(activation)
        act = nn.ReLU if activation == "relu" else nn.GELU
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            act(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        if norm == "batch":
            self.norm = nn.BatchNorm1d(embed_dim)
            self.norm_aggregate = nn.BatchNorm1d(embed_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm(embed_dim)
            self.norm_aggregate = nn.LayerNorm(embed_dim)
        else:
            raise ValueError("Unsupported norm type.")
        self.dropout_aggregate = nn.Dropout(dropout)

    def forward(self, x_prior, x):
        x = self.dropout_aggregate(x)
        x = x_prior + x
        x = self.norm_aggregate(x)
        x = self.mlp(x) + x
        return self.norm(x)


class _GTSEdgeAttention(nn.Module):
    """Original dense triangular attention block from principled GTS."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.linears = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(5)]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)
        num_nodes_q = query.size(1)
        num_nodes_k = key.size(1)

        left_k, right_k, left_v, right_v = [
            lin(t) for lin, t in zip(self.linears, (query, key, value, value))
        ]
        left_k = left_k.view(
            num_batches, num_nodes_q, num_nodes_q, self.num_heads, self.d_k
        )
        right_k = right_k.view(
            num_batches, num_nodes_k, num_nodes_k, self.num_heads, self.d_k
        )
        left_v = left_v.view_as(right_k)
        right_v = right_v.view_as(right_k)

        scores = torch.einsum("bxahd,bayhd->bxayh", left_k, right_k) / math.sqrt(
            self.d_k
        )
        if mask is not None:
            scores_dtype = scores.dtype
            scores = (
                scores.to(torch.float32)
                .masked_fill(mask.unsqueeze(4), -1e9)
                .to(scores_dtype)
            )

        att = F.softmax(scores, dim=2)
        att = self.dropout(att)
        val = torch.einsum("bxahd,bayhd->bxayhd", left_v, right_v)
        x = torch.einsum("bxayh,bxayhd->bxyhd", att, val)
        x = x.view(num_batches, num_nodes_q, num_nodes_k, self.embed_dim)
        return self.linears[-1](x)


class _GTSEdgeTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        activation: str = "relu",
        norm: str = "layer",
        norm_first: bool = True,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.attention = _GTSEdgeAttention(embed_dim, num_heads, attention_dropout)
        if norm_first:
            self.norm = nn.LayerNorm(embed_dim)
        self.ffn = _GTSFFN(embed_dim, dropout, activation, norm)

    def forward(self, x_in, mask=None):
        x = self.norm(x_in) if self.norm_first else x_in
        x_upd = self.attention(x, x, x, ~mask if mask is not None else None)
        return self.ffn(x_in, x_upd)


class _GTSDecomposer(nn.Module):
    def __init__(self, embed_dim: int, reduce_fn: str = "sum"):
        super().__init__()
        self.node_dim = embed_dim
        self.reduce_fn = reduce_fn
        self.out_proj = _GTSMLP(embed_dim, 2 * embed_dim)
        self.node_mlp = _GTSMLP(embed_dim, embed_dim)

    def forward(self, x, node_features, node_batch, token_index):
        x = self.out_proj(x)
        dim_size = node_batch.size(0)
        node_features = torch.zeros_like(node_features)
        for i in range(2):
            features_i = x[:, i * self.node_dim : (i + 1) * self.node_dim]
            features_i = scatter(
                features_i, token_index[i], 0, dim_size=dim_size, reduce=self.reduce_fn
            )
            node_features = node_features + features_i
        return self.node_mlp(node_features)


def _gts_apply_mask_2d(node_features, node_batch):
    _, mask = to_dense_batch(node_features, node_batch)
    unbatch = mask.unsqueeze(2) * mask.unsqueeze(1)
    tri_mask = unbatch.unsqueeze(3) * mask.unsqueeze(1).unsqueeze(2)
    return unbatch, tri_mask


class _GTSHead(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int, activation: str = "relu"):
        super().__init__()
        activation = _normalize_gts_activation(activation)
        act_fn = nn.ReLU if activation == "relu" else nn.GELU
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            act_fn(),
            nn.Dropout(0.0),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            act_fn(),
            nn.Dropout(0.0),
            nn.Linear(embed_dim // 4, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class _GTSEdgeTransformerNodeModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        num_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        activation: str,
    ):
        super().__init__()
        activation = _normalize_gts_activation(activation)
        self.feature_encoder = _GTSFeatureEncoder(hidden_dim, in_dim, edge_dim)
        self.composer = _GTSComposer(hidden_dim)
        self.layers = nn.ModuleList(
            [
                _GTSEdgeTransformerLayer(
                    hidden_dim,
                    num_heads,
                    ffn_dropout,
                    attention_dropout,
                    activation=activation,
                    norm="layer",
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.decomposer = _GTSDecomposer(hidden_dim)
        self.head = _GTSHead(hidden_dim, out_dim, activation=activation)

    def forward(self, data):
        data = self.feature_encoder(data)
        token_index = data.token_index
        x = self.composer(
            data.x, data.edge_index, data.edge_attr, data.batch, token_index, None
        )
        unbatch, mask = _gts_apply_mask_2d(data.x, data.batch)
        for layer in self.layers:
            x = layer(x, mask)
        x = x[unbatch]
        x = self.decomposer(x, data.x, data.batch, token_index)
        return self.head(x)


class TriangularTransformerNodeRegressor(nn.Module):
    """Self-contained original GTS triangular-attention transformer."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        out_dim: int = 1,
        num_heads: int = 4,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.model = _GTSEdgeTransformerNodeModel(
            in_dim=in_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_dim=out_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            activation=activation,
        )

    @staticmethod
    def _build_token_index(batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=batch.device)
        num_graphs = int(batch.max().item()) + 1
        parts = []
        for graph_idx in range(num_graphs):
            node_idx = (batch == graph_idx).nonzero(as_tuple=False).view(-1)
            n_nodes = int(node_idx.numel())
            if n_nodes == 0:
                continue
            src = node_idx.repeat_interleave(n_nodes)
            dst = node_idx.repeat(n_nodes)
            parts.append(torch.stack([src, dst], dim=0))
        if not parts:
            return torch.zeros((2, 0), dtype=torch.long, device=batch.device)
        return torch.cat(parts, dim=1)

    def forward(self, data) -> torch.Tensor:
        if getattr(data, "batch", None) is None:
            data.batch = data.x.new_zeros(data.x.size(0), dtype=torch.long)
        token_index = getattr(data, "token_index", None)
        if token_index is None or token_index.numel() == 0:
            data.token_index = self._build_token_index(data.batch)
        return self.model(data)


def build_transformer_model_from_dataset(
    dataset,
    hidden_dim: int = 64,
    num_layers: int = 10,
    out_dim: int = 1,
    num_heads: int = 8,
    attention_dropout: float = 0.2,
    ffn_dropout: float = 0.0,
    activation: str = "gelu",
) -> TriangularTransformerNodeRegressor:
    """Build the GTS triangular transformer (dense O(N^3); prefer GPU)."""
    sample = dataset[0]
    in_dim = sample.x.size(-1)
    edge_dim = sample.edge_attr.size(-1)
    return TriangularTransformerNodeRegressor(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=out_dim,
        num_heads=num_heads,
        attention_dropout=attention_dropout,
        ffn_dropout=ffn_dropout,
        activation=activation,
    )

