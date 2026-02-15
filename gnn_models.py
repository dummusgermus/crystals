from __future__ import annotations

import importlib.util
import os
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


class EdgeFNNHadamardConv(MessagePassing):
    """Implements: f^{t+1}(v) = FNN(sigma(f^t(v) W1 + sum_w FNN1(...))) ⊙ sum_w FNN2(...)."""

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
        self.message_mlp_1 = MLP(
            in_dim * 2 + edge_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            num_layers=2,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )
        self.message_mlp_2 = MLP(
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
        agg_1, agg_2 = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        base = self.update_mlp(self.act(x_proj + agg_1))
        gate = torch.sigmoid(agg_2)
        return base * gate

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp_1(msg_input), self.message_mlp_2(msg_input)

    def aggregate(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        msg_1, msg_2 = inputs
        agg_1 = self.aggr_module(msg_1, index, ptr=ptr, dim_size=dim_size)
        agg_2 = self.aggr_module(msg_2, index, ptr=ptr, dim_size=dim_size)
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size > 0:
            deg = torch.bincount(index, minlength=dim_size).clamp(min=1).to(agg_2.device)
            agg_2 = agg_2 / deg.view(-1, 1)
        return agg_1, agg_2


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


class GNNGraphRegressorHadamard(nn.Module):
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
                EdgeFNNHadamardConv(
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

        graph_repr = self.pool(x, batch)
        return self.output_mlp(graph_repr)


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


class GNNNodeRegressorHadamard(nn.Module):
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
                EdgeFNNHadamardConv(
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


def _load_principled_gts_module():
    module_path = os.path.join(
        os.path.dirname(__file__),
        "towards-principled-gts-main",
        "edge_transformer.py",
    )
    if not os.path.exists(module_path):
        raise FileNotFoundError(
            "Expected edge transformer source at "
            f"{module_path}. Please ensure towards-principled-gts-main is present."
        )
    spec = importlib.util.spec_from_file_location("principled_gts_edge_transformer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TriangularTransformerNodeRegressor(nn.Module):
    """Wrapper around principled GTS edge transformer for node-level prediction."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        out_dim: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        gts = _load_principled_gts_module()
        gts_activation = activation.lower()
        if gts_activation not in {"relu", "gelu"}:
            gts_activation = "relu"

        feature_encoder = gts.FeatureEncoder(
            embed_dim=hidden_dim,
            node_encoder="linear",
            edge_encoder="linear",
            node_dim=in_dim,
            edge_dim=edge_dim,
        )
        self.model = gts.EdgeTransformer(
            feature_encoder=feature_encoder,
            num_layers=num_layers,
            embed_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            activation=gts_activation,
            pooling=None,  # node-level output
            attention_dropout=dropout,
            ffn_dropout=dropout,
            has_edge_attr=True,
            compiled=False,
        )

    @staticmethod
    def _build_token_index(batch: torch.Tensor) -> torch.Tensor:
        if batch.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=batch.device)
        num_graphs = int(batch.max().item()) + 1
        parts = []
        for graph_idx in range(num_graphs):
            node_idx = (batch == graph_idx).nonzero(as_tuple=False).view(-1)
            n = int(node_idx.numel())
            if n == 0:
                continue
            src = node_idx.repeat_interleave(n)
            dst = node_idx.repeat(n)
            parts.append(torch.stack([src, dst], dim=0))
        if not parts:
            return torch.zeros((2, 0), dtype=torch.long, device=batch.device)
        return torch.cat(parts, dim=1)

    def forward(self, data) -> torch.Tensor:
        if getattr(data, "batch", None) is None:
            data.batch = data.x.new_zeros(data.x.size(0), dtype=torch.long)
        data.token_index = self._build_token_index(data.batch)
        return self.model(data)


def build_model_from_dataset(
    dataset,
    hidden_dim: int = 128,
    num_layers: int = 3,
    out_dim: int = 1,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    activation: str = "silu",
) -> GNNNodeRegressor:
    sample = dataset[0]
    in_dim = sample.x.size(-1)
    edge_dim = sample.edge_attr.size(-1)
    return GNNNodeRegressor(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=out_dim,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation,
    )


def build_hadamard_model_from_dataset(
    dataset,
    hidden_dim: int = 128,
    num_layers: int = 3,
    out_dim: int = 1,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    activation: str = "silu",
) -> GNNNodeRegressorHadamard:
    sample = dataset[0]
    in_dim = sample.x.size(-1)
    edge_dim = sample.edge_attr.size(-1)
    return GNNNodeRegressorHadamard(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=out_dim,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation,
    )


def build_transformer_model_from_dataset(
    dataset,
    hidden_dim: int = 128,
    num_layers: int = 3,
    out_dim: int = 1,
    num_heads: int = 4,
    dropout: float = 0.0,
    activation: str = "relu",
) -> TriangularTransformerNodeRegressor:
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
        dropout=dropout,
        activation=activation,
    )


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
