"""Dataset variants ("views") of the displacement superset graphs.

A view spec is a dict (or a compact string, see parse_spec), e.g.
    "rt=8.9,buf=4,edge=k3,feat=base+force,head=eqv"

keys
  rt     point only: target radius (A). Atoms with dist <= rt carry the loss; 0 = all atoms.
  buf    point only: extra context shell (A) around the target region (no loss there).
  edge   message-passing edges: kN = neighbour shells 1..N of the centre atom, rX = radius X A.
  hedge  edges used by the vector output head (default = edge).
  feat   '+'-joined node/edge feature groups:
           base   type, initial PE, is_defect, dist to defect          (= production x[:4])
           cyc    3-/4-cycle counts (3-shell graph, log1p)               (= production x[4:6])
           force  |F0| per atom; F0_i.u_ij and F0_j.u_ij per edge (initial forces, eV/A)
           dvec   cosine between bond u_ij and direction from the defect to atom i
           dirs   raw Cartesian components (bond vector, dvec, F0) -> only for the
                  non-equivariant 'direct' head control
  stable 1 = drop structures flagged unstable from TRAINING (evaluation always reports both)

Graph layout produced (PyG Data):
  x, z, edge_index (row 0 = neighbour j, row 1 = centre i: messages j -> i),
  edge_attr (col 0 = distance, as expected by atom_encodings / edge_features),
  edge_u (unit vector i -> j), hedge_index / hedge_u / hedge_attr (head edges),
  force0, y (= displacement, A), loss_mask (bool), unstable, cell_y (planar: c-vector change).
"""
from __future__ import annotations

import math

import torch
from torch_geometric.data import Data

DEFAULT = dict(rt=8.9, buf=4.0, edge="k3", hedge="", feat="base+cyc", stable=0)


def parse_spec(s) -> dict:
    if isinstance(s, dict):
        out = dict(DEFAULT); out.update(s); return out
    out = dict(DEFAULT)
    for part in filter(None, str(s).split(",")):
        k, v = part.split("=", 1)
        out[k] = v if k in ("edge", "hedge", "feat") else float(v)
    return out


def spec_key(spec: dict) -> str:
    return "rt{rt:g}_buf{buf:g}_{edge}_h{h}_{feat}_st{stable:g}".format(h=spec["hedge"] or spec["edge"], **spec)


def _edge_mask(g, rule: str):
    if rule.startswith("k"):
        return g.edge_shell.long() <= int(rule[1:])
    if rule.startswith("r"):
        return g.edge_dist <= float(rule[1:]) + 1e-6
    raise ValueError(rule)


def make_view(g, spec: dict) -> Data:
    point = g.domain == "point"
    feats = set(spec["feat"].split("+"))
    # ---- nodes
    full = bool(getattr(g, "full_cell", False))
    if full and "cyc" in feats:
        raise ValueError("full-cell point graphs have no cycle features")
    if point and spec["rt"] > 0:
        keep = torch.ones(g.num_nodes, dtype=torch.bool) if full else g.dist <= spec["rt"] + spec["buf"] + 1e-6
        loss = g.dist <= spec["rt"] + 1e-6
    else:
        keep = torch.ones(g.num_nodes, dtype=torch.bool)
        loss = torch.ones(g.num_nodes, dtype=torch.bool)
    idx = torch.nonzero(keep).view(-1)
    remap = torch.full((g.num_nodes,), -1, dtype=torch.long); remap[idx] = torch.arange(len(idx))
    ctr, nb = g.edge_index.long()  # stored as (centre i -> neighbour j)
    inside = keep[ctr] & keep[nb]

    def edges(rule):
        m = inside & _edge_mask(g, rule)
        c, n = remap[ctr[m]], remap[nb[m]]
        vec = g.edge_vec[m]; d = g.edge_dist[m]
        return c, n, vec, d, m

    c, n, vec, d, m = edges(spec["edge"])
    u = vec / d.unsqueeze(-1)
    F = g.force0[idx]; dv = g.dvec[idx]
    # ---- node features
    cols = [g.type[idx].float().unsqueeze(-1), g.pe0[idx].unsqueeze(-1), g.is_defect[idx].unsqueeze(-1),
            g.dist[idx].unsqueeze(-1)]
    if "cyc" in feats:
        cols.append(torch.log1p(g.cyc_k3[idx]))
    if "force" in feats:
        cols.append(F.norm(dim=-1, keepdim=True))
    if "dirs" in feats:
        cols += [dv, F]
    x = torch.cat(cols, dim=-1)
    # ---- edge features (col 0 = distance)
    ecols = [d.unsqueeze(-1), (g.type[idx][c] == g.type[idx][n]).float().unsqueeze(-1),
             ((g.is_defect[idx][c] > 0) | (g.is_defect[idx][n] > 0)).float().unsqueeze(-1)]
    if "force" in feats:
        ecols += [(F[c] * u).sum(-1, keepdim=True), (F[n] * u).sum(-1, keepdim=True)]
    if "dvec" in feats:
        dn = dv[c].norm(dim=-1, keepdim=True)
        ecols.append(torch.where(dn > 1e-6, (dv[c] * u).sum(-1, keepdim=True) / dn.clamp(min=1e-6), torch.zeros_like(dn)))
    if "dirs" in feats:
        ecols.append(u)
    eattr = torch.cat(ecols, dim=-1)
    out = Data(x=x, z=g.z[idx], edge_index=torch.stack([n, c]), edge_attr=eattr, edge_u=u,
               force0=F, y=g.disp[idx], loss_mask=loss[idx], num_nodes=len(idx))
    if spec["hedge"] and spec["hedge"] != spec["edge"]:
        hc, hn, hvec, hd, _ = edges(spec["hedge"])
        out.hedge_index = torch.stack([hn, hc]); out.hedge_u = hvec / hd.unsqueeze(-1)
        out.hedge_dist = hd.unsqueeze(-1)
    out.dist = g.dist[idx]
    out.unstable = torch.tensor([bool(g.unstable)])
    out.null = torch.tensor([bool(g.disp[idx][loss[idx]].norm(dim=-1).max() < 1e-4)])
    if not point:
        # change of the c vector, components along the (initial) a, b, c directions, in A
        out.cell_y = g.cell_change_frac.view(1, 3) * g.cell0[0].norm(dim=-1).view(1, 3)
    else:
        out.cell_y = torch.zeros(1, 3)
    out.folder = g.folder
    out.key = getattr(g, "defect_key", getattr(g, "stack_sequence", ""))
    out.element_a, out.element_b = g.element_a, g.element_b
    return out
