"""Edge-distance representations (architecture sweep, item 1).

The models receive the bond length as one raw scalar in ``edge_attr[:, 0]``.
Materials GNNs (SchNet, CGCNN paper, MEGNet) usually expand it into a set of
Gaussian radial basis functions so the network can learn sharp, non-monotonic
responses to the distance.

Specs
-----
raw      : unchanged (default)
rbf{K}   : keep the raw distance and append K Gaussians
           exp(-(d - mu_k)^2 / (2 sigma^2)), mu_k evenly spaced on [0, R_MAX],
           sigma = spacing. R_MAX = 7 A covers all edges (max 6.4 A).
"""
from __future__ import annotations

import re

import torch

R_MAX = 7.0


def parse_edge(spec: str) -> int:
    if spec in ("", "raw", None):
        return 0
    m = re.fullmatch(r"rbf(\d+)", spec)
    if not m:
        raise ValueError(f"unknown edge spec {spec!r}")
    return int(m.group(1))


def rbf_expand(dist: torch.Tensor, k: int) -> torch.Tensor:
    mu = torch.linspace(0.0, R_MAX, k, dtype=dist.dtype, device=dist.device)
    sigma = R_MAX / (k - 1)
    return torch.exp(-((dist.unsqueeze(-1) - mu) ** 2) / (2 * sigma ** 2))


def encode_edges_graph(g, spec: str):
    k = parse_edge(spec)
    if k == 0:
        return g
    g = g.clone()
    g.edge_attr = torch.cat([g.edge_attr, rbf_expand(g.edge_attr[:, 0], k).to(g.edge_attr.dtype)], dim=-1)
    return g


def encode_edges(dataset, spec: str):
    if parse_edge(spec) == 0:
        return dataset
    return [encode_edges_graph(g, spec) for g in dataset]
