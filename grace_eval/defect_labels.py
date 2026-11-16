"""Geometric planar-fault labels for new structures.

Training labels (laves_defect_atoms_c14c15.json) mark atoms where the stacking
deviates from pure C14 or C15. They are keyed by atom ids of our own cell
construction, so they cannot be transferred to foreign cells. Here an atom is a
defect atom if its local environment (own type + multiset of (normalised edge
length, same-type) over its graph edges) occurs neither in pure C14 nor in pure
C15 of the same compound."""
from collections import Counter
import torch


def fingerprints(g, nd=2):
    ei, ea = g.edge_index, g.edge_attr
    dmin = float(ea[:, 0].min())
    nb = [[] for _ in range(g.num_nodes)]
    for (i, j), a in zip(ei.t().tolist(), ea.tolist()):
        key = (round(a[0] / dmin, nd), int(a[1]))
        nb[i].append(key); nb[j].append(key)
    t = g.x[:, 0].long().tolist()
    return [(t[i], tuple(sorted(nb[i]))) for i in range(g.num_nodes)]


def defect_mask(g, refs, nd=2):
    ref = set()
    for r in refs:
        ref |= set(fingerprints(r, nd))
    return torch.tensor([fp not in ref for fp in fingerprints(g, nd)])


def dist_to_defect(g, mask, cell=None):
    """Same definition as planar_graph_maker: min-image distance to nearest defect atom (0 if none)."""
    import numpy as np
    n = g.num_nodes
    if not mask.any():
        return torch.zeros(n)
    pos = g.pos.double().numpy()
    if cell is None:
        raise ValueError("cell needed")
    inv = np.linalg.inv(cell); frac = pos @ inv
    d = torch.zeros(n)
    idx = np.where(mask.numpy())[0]
    for i in range(n):
        if mask[i]:
            continue
        df = frac[idx] - frac[i]; df -= np.round(df)
        d[i] = float(np.linalg.norm(df @ cell, axis=1).min())
    return d
