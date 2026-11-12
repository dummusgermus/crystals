"""Positional / structural encodings for the defect graphs (PE sweep).

The current models use no positional encodings: the Transformer attends over all
atoms with a learned bias on k-NN edges only, CGCNN only sees edge distances.
This module adds two families, computed once per graph from its (undirected)
k-NN edge graph:

Node-level (concatenated to ``x``; usable by CGCNN and Transformer)
    rwse{k}  random-walk structural encoding: return probabilities diag(P^i),
             i = 1..k, P = D^-1 A.
    lap{k}   Laplacian PE: k lowest non-trivial eigenvectors of the normalised
             Laplacian (zero-padded for tiny graphs). Sign is ambiguous ->
             random per-graph sign flips during training (``lap_sign_flip``).

Pair-level (Transformer only; additive per-head attention bias over ALL pairs)
    spd      shortest-path hop distance (Graphormer-style embedding).
    geo      geodesic bond-path length in Angstrom (shortest path weighted by
             edge distance), RBF-expanded. A PBC-safe proxy for the all-pair
             distance: the raw LAMMPS cells are not available, and ``data.pos``
             is wrapped, so Euclidean distances between non-bonded atoms would
             be wrong.

Pair matrices are stored per graph as rows padded to ``pad`` columns
(``pair_spd`` uint8, ``pair_geo`` float16, shape [N, pad]) so PyG batching
concatenates them along dim 0 like ``x``.

A PE spec is a "+"-joined list, e.g. ``"none"``, ``"rwse16"``, ``"lap8+rwse16"``,
``"spd+geo"``, ``"rwse16+spd+geo"``.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import to_dense_batch

from gnn_models import GraphTransformerNodeRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "pe_sweep", "cache")
MAX_RW = 16
MAX_LAP = 16
MAX_HOP = 14  # hop distances >= MAX_HOP share one bucket; unreachable = MAX_HOP + 1
GEO_CAP = 40.0  # Angstrom; unreachable pairs get this value


# ---------------------------------------------------------------------------
# spec parsing
# ---------------------------------------------------------------------------


def parse_pe(spec: str) -> Dict[str, int]:
    """'lap8+rwse16+spd' -> {'lap': 8, 'rwse': 16, 'spd': 1}."""
    out: Dict[str, int] = {}
    if spec in ("", "none"):
        return out
    for part in spec.split("+"):
        m = re.fullmatch(r"(rwse|lap)(\d+)", part)
        if m:
            out[m.group(1)] = int(m.group(2))
        elif part in ("spd", "geo"):
            out[part] = 1
        else:
            raise ValueError(f"Unknown PE component {part!r} in {spec!r}")
    if out.get("rwse", 0) > MAX_RW or out.get("lap", 0) > MAX_LAP:
        raise ValueError(f"rwse<= {MAX_RW}, lap <= {MAX_LAP}")
    return out


def has_pair_bias(spec: str) -> bool:
    p = parse_pe(spec)
    return "spd" in p or "geo" in p


# ---------------------------------------------------------------------------
# per-graph computation (cached per dataset)
# ---------------------------------------------------------------------------


def _adjacency(edge_index: torch.Tensor, edge_dist: torch.Tensor, n: int):
    s, t = edge_index.numpy()
    w = edge_dist.numpy().astype(np.float64)
    keep = s != t
    s, t, w = s[keep], t[keep], w[keep]
    # undirected, deduplicated (keep the shortest distance per pair)
    a, b = np.concatenate([s, t]), np.concatenate([t, s])
    ww = np.concatenate([w, w])
    order = np.lexsort((ww, b, a))
    a, b, ww = a[order], b[order], ww[order]
    first = np.ones(len(a), dtype=bool)
    first[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
    a, b, ww = a[first], b[first], ww[first]
    A = np.zeros((n, n), dtype=np.float64)
    A[a, b] = 1.0
    W = coo_matrix((ww, (a, b)), shape=(n, n)).tocsr()
    return A, W


def compute_graph_pe(data) -> Dict[str, np.ndarray]:
    n = int(data.num_nodes)
    A, W = _adjacency(data.edge_index, data.edge_attr[:, 0], n)
    deg = A.sum(1)

    # RWSE: diag(P^i), i = 1..MAX_RW
    P = A / np.maximum(deg, 1.0)[:, None]
    rw = np.zeros((n, MAX_RW), dtype=np.float32)
    M = np.eye(n)
    for i in range(MAX_RW):
        M = M @ P
        rw[:, i] = np.diag(M)

    # LapPE: normalised Laplacian eigenvectors, skip the trivial one(s)
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    L = np.eye(n) - dinv[:, None] * A * dinv[None, :]
    evals, evecs = np.linalg.eigh(L)
    n_comp = int((np.abs(evals) < 1e-8).sum()) or 1
    vecs = evecs[:, n_comp : n_comp + MAX_LAP]
    lap = np.zeros((n, MAX_LAP), dtype=np.float32)
    lap[:, : vecs.shape[1]] = vecs
    # deterministic sign convention (largest-|.| entry positive); training flips randomly
    idx = np.abs(lap).argmax(0)
    sgn = np.sign(lap[idx, np.arange(MAX_LAP)])
    lap *= np.where(sgn == 0, 1.0, sgn)[None, :]

    hop = shortest_path(A, method="D", unweighted=True)
    hop = np.where(np.isinf(hop), MAX_HOP + 1, np.minimum(hop, MAX_HOP)).astype(np.uint8)
    geo = shortest_path(W, method="D", directed=False)
    geo = np.where(np.isinf(geo), GEO_CAP, np.minimum(geo, GEO_CAP)).astype(np.float16)
    return {"rwse": rw, "lap": lap, "spd": hop, "geo": geo}


def _cache_path(domain: str) -> str:
    return os.path.join(CACHE_DIR, f"pe_{domain}.pt")


def load_or_build_cache(dataset, domain: str) -> List[Dict[str, np.ndarray]]:
    path = _cache_path(domain)
    if os.path.isfile(path):
        cache = torch.load(path, weights_only=False)
        if len(cache) == len(dataset) and all(
            c["rwse"].shape[0] == int(d.num_nodes) for c, d in zip(cache, dataset)
        ):
            return cache
    cache = [compute_graph_pe(d) for d in dataset]
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(cache, tmp)
    os.replace(tmp, path)
    return cache


def apply_pe(dataset, cache, spec: str) -> Tuple[list, int]:
    """Return (new dataset, pad width). Node PEs are appended to x (rwse first,
    then lap, so lap columns are the last ``k_lap`` columns of x)."""
    p = parse_pe(spec)
    pad = max(int(d.num_nodes) for d in dataset)
    out = []
    for d, c in zip(dataset, cache):
        g = d.clone()
        extra = []
        if "rwse" in p:
            extra.append(torch.from_numpy(c["rwse"][:, : p["rwse"]]))
        if "lap" in p:
            extra.append(torch.from_numpy(c["lap"][:, : p["lap"]]))
        if extra:
            g.x = torch.cat([g.x] + [e.to(g.x.dtype) for e in extra], dim=-1)
        n = int(g.num_nodes)
        if "spd" in p:
            m = np.full((n, pad), MAX_HOP + 1, dtype=np.uint8)
            m[:, :n] = c["spd"]
            g.pair_spd = torch.from_numpy(m)
        if "geo" in p:
            m = np.full((n, pad), GEO_CAP, dtype=np.float16)
            m[:, :n] = c["geo"]
            g.pair_geo = torch.from_numpy(m)
        out.append(g)
    return out, pad


def lap_sign_flip(batch, k_lap: int) -> None:
    """In-place random sign flip of the LapPE columns (last k_lap of x), per graph."""
    if k_lap <= 0:
        return
    n_graphs = int(batch.num_graphs)
    sign = torch.randint(0, 2, (n_graphs, k_lap), device=batch.x.device) * 2 - 1
    x = batch.x.clone()
    x[:, -k_lap:] = x[:, -k_lap:] * sign[batch.batch].to(x.dtype)
    batch.x = x


# ---------------------------------------------------------------------------
# Transformer with all-pair attention bias
# ---------------------------------------------------------------------------


class GraphTransformerPairBias(GraphTransformerNodeRegressor):
    """GraphTransformerNodeRegressor + per-head additive bias for every atom pair
    from hop distance (embedding) and/or geodesic distance (RBF -> linear)."""

    def __init__(self, *args, use_spd: bool = False, use_geo: bool = False,
                 n_rbf: int = 32, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        h = self.num_heads
        self.use_spd, self.use_geo = use_spd, use_geo
        if use_spd:
            self.spd_emb = nn.Embedding(MAX_HOP + 2, h)
            nn.init.zeros_(self.spd_emb.weight)
        if use_geo:
            self.register_buffer("rbf_centers", torch.linspace(0.0, 25.0, n_rbf))
            self.rbf_gamma = 1.0 / (25.0 / n_rbf) ** 2
            self.geo_lin = nn.Linear(n_rbf, h)
            nn.init.zeros_(self.geo_lin.weight)
            nn.init.zeros_(self.geo_lin.bias)

    def _pair_bias(self, data, n_max: int) -> torch.Tensor:
        bias = 0.0
        if self.use_spd:
            rows, _ = to_dense_batch(data.pair_spd.long(), data.batch)  # (B, N, pad)
            bias = bias + self.spd_emb(rows[:, :, :n_max])
        if self.use_geo:
            rows, _ = to_dense_batch(data.pair_geo.float(), data.batch)
            d = rows[:, :, :n_max].unsqueeze(-1)
            rbf = torch.exp(-self.rbf_gamma * (d - self.rbf_centers) ** 2)
            bias = bias + self.geo_lin(rbf)
        return bias  # (B, N, N, heads)

    def forward(self, data) -> torch.Tensor:
        from torch_geometric.utils import to_dense_adj
        import torch.nn.functional as F

        if getattr(data, "batch", None) is None:
            data.batch = data.x.new_zeros(data.x.size(0), dtype=torch.long)
        x = self.node_encoder(data.x)
        edge_bias = self.edge_encoder(data.edge_attr)
        x_pad, node_mask = to_dense_batch(x, data.batch)
        n_max = x_pad.size(1)
        attn = to_dense_adj(data.edge_index, data.batch, edge_bias, max_num_nodes=n_max)
        attn = attn + self._pair_bias(data, n_max)
        pad = ~node_mask
        attn = attn.masked_fill(pad.unsqueeze(2).unsqueeze(-1), float("-inf"))
        attn = attn.masked_fill(pad.unsqueeze(1).unsqueeze(-1), float("-inf"))
        eye = torch.eye(n_max, device=attn.device, dtype=torch.bool)
        attn = attn.masked_fill(eye.view(1, n_max, n_max, 1), 0.0)
        attn_mask = attn.permute(0, 3, 1, 2).reshape(-1, n_max, n_max)
        out = self.encoder(x_pad, mask=attn_mask)
        if self.attention_dropout > 0:
            out = F.dropout(out, p=self.attention_dropout, training=self.training)
        return self.head(out[node_mask])


def build_pair_bias_transformer(dataset, config: dict, spec: str) -> GraphTransformerPairBias:
    p = parse_pe(spec)
    s = dataset[0]
    return GraphTransformerPairBias(
        in_dim=s.x.size(-1),
        edge_dim=s.edge_attr.size(-1),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
        activation=config["activation"],
        use_spd="spd" in p,
        use_geo="geo" in p,
    )
