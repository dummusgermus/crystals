from __future__ import annotations

import math
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


class _GTSMLP(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, linear: bool = False):
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
            data.edge_attr = torch.ones((data.edge_index.size(1), 1), device=data.x.device)
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
    def __init__(self, embed_dim: int, dropout: float = 0.0, activation: str = "relu", norm: str = "layer"):
        super().__init__()
        if activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU
        else:
            raise ValueError("Original GTS supports only relu/gelu activations.")
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
        self.linears = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(5)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)
        num_nodes_q = query.size(1)
        num_nodes_k = key.size(1)

        left_k, right_k, left_v, right_v = [l(x) for l, x in zip(self.linears, (query, key, value, value))]
        left_k = left_k.view(num_batches, num_nodes_q, num_nodes_q, self.num_heads, self.d_k)
        right_k = right_k.view(num_batches, num_nodes_k, num_nodes_k, self.num_heads, self.d_k)
        left_v = left_v.view_as(right_k)
        right_v = right_v.view_as(right_k)

        scores = torch.einsum("bxahd,bayhd->bxayh", left_k, right_k) / math.sqrt(self.d_k)
        if mask is not None:
            scores_dtype = scores.dtype
            scores = scores.to(torch.float32).masked_fill(mask.unsqueeze(4), -1e9).to(scores_dtype)

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
            features_i = scatter(features_i, token_index[i], 0, dim_size=dim_size, reduce=self.reduce_fn)
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
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError("Original GTS supports only relu/gelu activations.")
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
        dropout: float,
        activation: str,
    ):
        super().__init__()
        if activation not in {"relu", "gelu"}:
            raise ValueError("Original GTS uses relu or gelu activation.")
        self.feature_encoder = _GTSFeatureEncoder(hidden_dim, in_dim, edge_dim)
        self.composer = _GTSComposer(hidden_dim)
        self.layers = nn.ModuleList(
            [
                _GTSEdgeTransformerLayer(
                    hidden_dim,
                    num_heads,
                    dropout,
                    dropout,
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
        x = self.composer(data.x, data.edge_index, data.edge_attr, data.batch, token_index, None)
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
        dropout: float = 0.0,
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
            dropout=dropout,
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
