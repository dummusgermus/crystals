"""Displacement models: production backbones (gated CGCNN / GraphTransformer) + vector heads.

Backbones are the production classes from gnn_models.py with out_dim = hidden, so they
return a per-atom embedding h_i (rotation-invariant, since all inputs are invariant
unless the view uses the 'dirs' features).

Heads
  eqv     d_i = sum_j phi(h_i, h_j, rbf(r_ij)[, F_i.u_ij, F_j.u_ij]) * u_ij
          Rotation-equivariant and translation-invariant by construction; gives exactly
          zero on sites with inversion / cubic symmetry when the inputs are symmetric.
  eqvF    eqv + alpha(h_i) * F0_i   (initial force as an extra equivariant direction)
  direct  d_i = MLP(h_i) -> 3       (non-equivariant control; needs feat 'dirs')
Planar graphs also get a graph-level head for the change of the c vector (3 components
along the initial a, b, c directions; invariant numbers): MLP(mean_i h_i).
"""
from __future__ import annotations

import os
import sys

import torch
from torch import nn
from torch_geometric.utils import scatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from edge_features import rbf_expand  # noqa: E402
from gnn_models import GatedGNNNodeRegressor, GraphTransformerNodeRegressor  # noqa: E402


def mlp(i, h, o, act=nn.SiLU):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, h), act(), nn.Linear(h, o))


class DisplacementModel(nn.Module):
    def __init__(self, kind: str, in_dim: int, edge_dim: int, head: str = "eqv", hidden: int = 128,
                 num_layers: int = 4, use_force_in_head: bool = False, cell_head: bool = False,
                 num_heads: int = 4, dropout: float = 0.1, n_rbf: int = 16, norm: bool = False):
        super().__init__()
        self.kind, self.head, self.n_rbf = kind, head, n_rbf
        # norm: LayerNorm after the input MLP and after every gated conv (CGCNN) and on h before
        # the head -- the production CGCNN sums ~30 messages per layer without normalisation,
        # which lets activations grow geometrically on out-of-distribution compounds.
        self.norm = norm
        self.use_force = use_force_in_head
        if kind == "cgcnn":
            self.backbone = GatedGNNNodeRegressor(in_dim, edge_dim, hidden_dim=hidden, num_layers=num_layers,
                                                  out_dim=hidden, activation="silu")
        elif kind == "transformer":
            self.backbone = GraphTransformerNodeRegressor(in_dim, edge_dim, hidden_dim=hidden, num_layers=num_layers,
                                                          out_dim=hidden, num_heads=num_heads, dropout=dropout,
                                                          attention_dropout=dropout)
        else:
            raise ValueError(kind)
        if head in ("eqv", "eqvF"):
            self.phi = mlp(2 * hidden + n_rbf + (2 if use_force_in_head else 0), hidden, 1)
            if head == "eqvF":
                self.alpha = mlp(hidden, hidden, 1)
        elif head == "direct":
            self.out = mlp(hidden, hidden, 3)
        else:
            raise ValueError(head)
        self.cell = mlp(hidden, hidden, 3) if cell_head else None
        if norm:
            n_conv = num_layers if kind == "cgcnn" else 0
            self.lns = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_conv + 1))
            self.ln_h = nn.LayerNorm(hidden)

    def _backbone(self, data):
        if not self.norm:
            return self.backbone(data)
        if self.kind == "cgcnn":
            bb = self.backbone
            x = self.lns[0](bb.input_mlp(data.x))
            for conv, ln in zip(bb.convs, self.lns[1:]):
                x = ln(conv(x, data.edge_index, data.edge_attr))
            return self.ln_h(bb.output_mlp(x))
        return self.ln_h(self.backbone(data))

    def forward(self, data):
        h = self._backbone(data)
        if self.head == "direct":
            d = self.out(h)
        else:
            if hasattr(data, "hedge_index"):
                (nb, ctr), u, dist = data.hedge_index, data.hedge_u, data.hedge_dist.view(-1)
            else:
                (nb, ctr), u, dist = data.edge_index, data.edge_u, data.edge_attr[:, 0]
            z = [h[ctr], h[nb], rbf_expand(dist, self.n_rbf)]
            if self.use_force:
                z += [(data.force0[ctr] * u).sum(-1, keepdim=True), (data.force0[nb] * u).sum(-1, keepdim=True)]
            w = self.phi(torch.cat(z, -1))
            d = scatter(w * u, ctr, 0, dim_size=h.size(0), reduce="sum")
            if self.head == "eqvF":
                d = d + self.alpha(h) * data.force0
        cell = None
        if self.cell is not None:
            batch = data.batch if getattr(data, "batch", None) is not None else h.new_zeros(h.size(0), dtype=torch.long)
            cell = self.cell(scatter(h, batch, 0, reduce="mean"))
        return d, cell


class PaiNNLayer(nn.Module):
    """Message + update block of PaiNN (Schuett et al. 2021) with scalar s and vector v features."""

    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.phi = mlp(hidden, hidden, 3 * hidden)
        self.W = nn.Linear(edge_dim, 3 * hidden)
        self.U = nn.Linear(hidden, hidden, bias=False)
        self.V = nn.Linear(hidden, hidden, bias=False)
        self.upd = mlp(2 * hidden, hidden, 3 * hidden)
        self.h = hidden

    def forward(self, s, v, nb, ctr, u, e):
        H = self.h
        x = self.phi(s)[nb] * self.W(e)
        a, b, c = x.split(H, -1)
        dv = v[nb] * a.unsqueeze(1) + b.unsqueeze(1) * u.unsqueeze(-1)
        s = s + scatter(c, ctr, 0, dim_size=s.size(0), reduce="sum")
        v = v + scatter(dv, ctr, 0, dim_size=s.size(0), reduce="sum")
        Uv, Vv = self.U(v), self.V(v)
        y = self.upd(torch.cat([s, torch.sqrt((Vv ** 2).sum(1) + 1e-8)], -1))
        aa, bb, cc = y.split(H, -1)
        v = v + aa.unsqueeze(1) * Uv
        s = s + bb * (Uv * Vv).sum(1) + cc
        return s, v


class PaiNNDisplacement(nn.Module):
    """Equivariant backbone: scalars from the node features, vectors start at zero (or at
    the initial force); output d_i = Linear(v_i) (vector channel mix) [+ eqv edge head]."""

    def __init__(self, in_dim: int, edge_dim: int, hidden: int = 128, num_layers: int = 4,
                 force_init: bool = False, cell_head=False, norm: bool = False, zemb: bool = False,
                 energy_head: bool = False, e_coupling: bool = False):
        super().__init__()
        # cell_head: True = 3 invariant c-vector components (planar); "tensor" = equivariant 3x3 T with
        #   relaxed cell = cell0 @ (I + T) (bulk benchmarks): node term sum_c w_c(s) v_c (x) U v_c, edge term
        #   phi_ij u_ij (x) u_ij, isotropic a(s) I and b(s) * initial stress (when data.stress0 exists).
        # zemb: learned element embedding added to the scalar features (multi-element benchmarks).
        # norm: LayerNorm on the scalars and an equivariant RMS norm on the vectors after
        # every layer (v / sqrt(mean_c |v_c|^2), learnable per-channel gain).
        self.norm = norm
        if norm:
            self.ln_s = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(num_layers))
            self.g_v = nn.ParameterList(nn.Parameter(torch.ones(hidden)) for _ in range(num_layers))
        self.embed = mlp(in_dim, hidden, hidden)
        self.layers = nn.ModuleList(PaiNNLayer(hidden, edge_dim) for _ in range(num_layers))
        self.out = nn.Linear(hidden, 1, bias=False)
        self.force_init = force_init
        self.f0 = nn.Linear(1, hidden, bias=False) if force_init else None
        self.tensor_head = cell_head == "tensor"
        if self.tensor_head:
            self.cell = None
            self.t_U = nn.Linear(hidden, hidden, bias=False); self.t_w = nn.Linear(hidden, hidden)
            self.t_phi = mlp(hidden, hidden, hidden); self.t_e = nn.Linear(edge_dim, hidden)
            self.t_phi_out = nn.Linear(hidden, 1)
            self.t_iso = mlp(hidden, hidden, 2)
        else:
            self.cell = mlp(hidden, hidden, 3) if cell_head else None
        self.zemb = nn.Embedding(119, hidden) if zemb else None
        # energy head (multi-task): per-atom relaxation energy e_i (target units), stored in self.e_atom.
        # e_coupling: + a * F0_i . d_i (the harmonic estimate -1/2 F.d, from the PREDICTED displacement)
        self.e_out = mlp(hidden, hidden, 1) if energy_head else None
        self.e_a = nn.Parameter(torch.zeros(1)) if e_coupling else None
        self.e_atom = None

    def _tensor(self, s, v, nb, ctr, u, e, batch, data):
        B = int(batch.max()) + 1 if batch.numel() else 1
        H = s.size(1)
        tn = torch.einsum("nah,nbh,nh->nab", v, self.t_U(v), self.t_w(s)) / H
        T = scatter(tn, batch, 0, dim_size=B, reduce="mean")
        phi = self.t_phi_out(self.t_phi(s)[ctr] * self.t_e(e))  # (E,1)
        te = phi.unsqueeze(-1) * u.unsqueeze(-1) * u.unsqueeze(-2)
        T = T + scatter(te, batch[ctr], 0, dim_size=B, reduce="mean")
        ab = self.t_iso(scatter(s, batch, 0, dim_size=B, reduce="mean"))
        T = T + ab[:, 0, None, None] * torch.eye(3, device=s.device)
        if getattr(data, "stress0", None) is not None:
            T = T + ab[:, 1, None, None] * 100.0 * data.stress0.view(B, 3, 3)
        return T

    def forward(self, data):
        s = self.embed(data.x)
        if self.zemb is not None:
            s = s + self.zemb(data.z.long())
        nb, ctr = data.edge_index
        u = data.edge_u
        v = s.new_zeros(s.size(0), 3, s.size(1))
        if self.force_init:
            v = v + self.f0(data.force0.unsqueeze(-1))  # (N,3,H)
        for i, layer in enumerate(self.layers):
            s, v = layer(s, v, nb, ctr, u, data.edge_attr)
            if self.norm:
                s = self.ln_s[i](s)
                v = v / torch.sqrt((v ** 2).sum(1, keepdim=True).mean(-1, keepdim=True) + 1e-6) * self.g_v[i]
        d = self.out(v).squeeze(-1)
        if self.e_out is not None:
            e = self.e_out(s).squeeze(-1)
            if self.e_a is not None:
                e = e + self.e_a * (data.force0 * d).sum(-1)
            self.e_atom = e
        cell = None
        if self.tensor_head:
            batch = data.batch if getattr(data, "batch", None) is not None else s.new_zeros(s.size(0), dtype=torch.long)
            return d, self._tensor(s, v, nb, ctr, u, data.edge_attr, batch, data)
        if self.cell is not None:
            batch = data.batch if getattr(data, "batch", None) is not None else s.new_zeros(s.size(0), dtype=torch.long)
            cell = self.cell(scatter(s, batch, 0, reduce="mean"))
        return d, cell
