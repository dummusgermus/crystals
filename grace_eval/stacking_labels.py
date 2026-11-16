"""Planar-fault labels in the training convention (laves_defect_atoms_c14c15.json),
derived from geometry so they can be applied to foreign cells.

A hexagonal Laves cell is a stack of units (B kagome layer + A-B-A triple layer).
Each triple layer gets a lateral descriptor (in-plane site class 0/1/2 of its
A, B, A atoms). The step from one unit to the next is a permutation of these
classes: a cyclic shift (+1 or -1, i.e. cubic/C15-like stacking), a transposition
(hexagonal/C14-like) or the identity. A unit's triple-layer atoms are labelled as
defect atoms if exactly one of its two steps (to the previous and to the next unit,
periodic) is hexagonal, i.e. if it sits at a boundary between C14-like and
C15-like stacking.
Validated to reproduce the training labels exactly (see validate())."""
import numpy as np


def _layers(pos, types, tol=0.25):
    z = pos[:, 2]; order = np.argsort(z)
    out, cur = [], [order[0]]
    for i in order[1:]:
        if z[i] - z[cur[-1]] > tol:
            out.append(cur); cur = [i]
        else:
            cur.append(i)
    out.append(cur)
    return [("".join(sorted("AB"[int(types[i]) - 1] for i in l)), list(l)) for l in out]


def unit_descriptors(pos, types, cell=None):
    L = _layers(pos, types)
    s = next(i for i, l in enumerate(L) if l[0] == "BBB")
    L = L[s:] + L[:s]
    if len(L) % 4:
        raise ValueError(f"{len(L)} layers, not a multiple of 4")
    if cell is not None:
        a1, a2 = np.asarray(cell[0][:2]), np.asarray(cell[1][:2])
    else:  # hexagonal cell inferred from the kagome layer (a = 2 * kagome nn distance)
        k = pos[L[0][1]]
        a = 2 * min(np.linalg.norm(k[i, :2] - k[j, :2]) for i in range(3) for j in range(i + 1, 3))
        a1, a2 = np.array([a, 0.0]), np.array([-a / 2, a * np.sqrt(3) / 2])
    inv = np.linalg.inv(np.stack([a1, a2]).T)

    def cls(p):
        f = inv @ p[:2]
        return int(round(3 * (f[0] - f[1]))) % 3

    units = []
    for u in range(len(L) // 4):
        kl, l1, l2, l3 = L[4 * u: 4 * u + 4]
        if (kl[0], l1[0], l2[0], l3[0]) != ("BBB", "A", "B", "A"):
            raise ValueError(f"unexpected layer pattern {[kl[0], l1[0], l2[0], l3[0]]}")
        d = (cls(pos[l1[1][0]]), cls(pos[l2[1][0]]), cls(pos[l3[1][0]]))
        units.append({"desc": d, "triple": l1[1] + l2[1] + l3[1]})
    return units


def step_kind(d0, d1):
    if len(set(d0)) != 3 or len(set(d1)) != 3:
        raise ValueError(f"degenerate descriptor {d0} -> {d1}")
    p = {a: b for a, b in zip(d0, d1)}
    if all(p[x] == x for x in range(3)):
        return "id"
    if all(p[x] == (x + 1) % 3 for x in range(3)):
        return "c+"
    if all(p[x] == (x - 1) % 3 for x in range(3)):
        return "c-"
    return "h"


def defect_mask(pos, types, cell=None):
    us = unit_descriptors(pos, types, cell)
    n = len(us)
    steps = [step_kind(us[i]["desc"], us[(i + 1) % n]["desc"]) for i in range(n)]
    mask = np.zeros(len(pos), dtype=bool)
    for i, u in enumerate(us):
        if (steps[i - 1] == "h") != (steps[i] == "h"):
            mask[u["triple"]] = True
    return mask, "".join({"c+": ">", "c-": "<", "h": "h", "id": "="}[s] for s in steps)


def validate(dataset_path="planar_pyg_dataset_residual_c14c15_totals.pt",
             labels_path="laves_defect_atoms_c14c15.json"):
    import json
    import torch
    ds = torch.load(dataset_path, weights_only=False)
    lab = json.load(open(labels_path))["defect_atoms"]
    ok = bad = 0
    for g in ds:
        m, _ = defect_mask(g.pos.numpy(), g.x[:, 0].numpy())
        truth = np.array([pid in set(lab[g.stack_sequence]) for pid in g.particle_ids.tolist()])
        ok += bool((m == truth).all()); bad += not bool((m == truth).all())
    return ok, bad
