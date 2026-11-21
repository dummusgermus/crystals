"""Run the authors' position-updating models (DefiNet, E3Relax, E3Relax-H2) on our displacement views.

The models update atom positions (and, for E3Relax/H2, the three lattice vectors) in every layer,
recomputing bond vectors from the updated geometry. This wrapper
  * builds their input namespace from a view batch (pos_u, cell_u, cell_offsets, neighbors, z, defect markers),
  * optionally adds our standardised node features to their atom embedding (feat_emb) and starts the
    vector channels from the initial force (f_init) -- the two hooks marked '# [disp-study]' in the vendored files,
  * converts every layer's (pos, cell) into our targets: displacement d (A) and, for planar faults,
    the non-affine displacement (fractional change x initial cell, mean removed) plus the c-vector change
    cell_y (components along the initial a, b, c in units of their lengths x lengths, as in views.py).
forward() returns (d / scale, cell_y / cell_scale) of the last layer, the same interface as models.py;
self.layers keeps (d / scale, cell / scale) of every layer for the layer-wise loss.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch_geometric.utils import scatter

from .definet_model import DefiNet
from .e3relax_model import E3Relax
from .e3relaxh2_model import E3RelaxH2

KINDS = {"definet": DefiNet, "e3relax": E3Relax, "e3relaxh2": E3RelaxH2}


class ExtRelax(nn.Module):
    def __init__(self, kind: str, feat_dim: int | None = None, force_init: bool = False, hidden: int = 512,
                 num_layers: int = 4, num_rbf: int = 128, cutoff: float = 30.0, planar: bool = False,
                 static_geom: bool = False, no_vn: bool = False):
        super().__init__()
        # authors' defaults: hidden 512, 4 layers, 128 Gaussian RBFs, RBF cutoff 30 A, 118 elements
        kw = dict(hidden_channels=hidden, num_layers=num_layers, num_rbf=num_rbf, cutoff=cutoff, num_elements=118)
        self.kind, self.planar = kind, planar
        self.net = KINDS[kind](**kw)
        self.net.static_geom, self.net.no_vn = static_geom, no_vn  # DefiNet ablations
        if feat_dim:
            self.net.feat_emb = nn.Linear(feat_dim, hidden)
        if force_init:
            self.net.f_init = nn.Linear(1, hidden, bias=False)
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("cell_scale", torch.tensor(1.0))
        self.layers = []

    def set_scales(self, scale, cell_scale):
        self.scale.fill_(float(scale)); self.cell_scale.fill_(float(cell_scale))

    def forward(self, data):
        batch = data.batch if getattr(data, "batch", None) is not None else \
            torch.zeros(data.num_nodes, dtype=torch.long, device=data.pos0.device)
        cell0 = data.cell0.view(-1, 3, 3)
        ns = SimpleNamespace(pos_u=data.pos0, cell_u=cell0, cell_offsets=data.cell_off, edge_index=data.edge_index,
                             neighbors=data.n_edges.view(-1), batch=batch, x=data.z.long(), defect=data.is_def.long(),
                             xf=data.x, force0=data.force0)
        pos_list, cell_list = self.net(ns)
        cinv0 = torch.linalg.inv(cell0)
        norms0 = cell0.norm(dim=-1)  # (B, 3)
        self.layers = []
        for pos, cell in zip(pos_list, cell_list):
            if self.planar:
                s = torch.einsum("np,npq->nq", pos, torch.linalg.inv(cell)[batch])
                s0 = torch.einsum("np,npq->nq", data.pos0, cinv0[batch])
                d = torch.einsum("nq,nqv->nv", s - s0, cell0[batch])
                d = d - scatter(d, batch, 0, reduce="mean")[batch]
                frac = torch.einsum("bv,bvq->bq", cell[:, 2] - cell0[:, 2], cinv0)
                cy = frac * norms0
            else:
                d = pos - data.pos0
                cy = torch.zeros(cell0.size(0), 3, device=d.device)
            self.layers.append((d / self.scale, cell / self.scale))
        return d / self.scale, (cy / self.cell_scale if self.planar else None)

    def cell_target(self, data):
        """Relaxed cell (A) per graph: initial cell with the c row changed by cell_y (planar)."""
        cell0 = data.cell0.view(-1, 3, 3)
        if not self.planar:
            return cell0
        frac = data.cell_y.view(-1, 3) / cell0.norm(dim=-1)
        dc = torch.einsum("bq,bqv->bv", frac, cell0)
        out = cell0.clone(); out[:, 2] = out[:, 2] + dc
        return out
