from __future__ import annotations

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

