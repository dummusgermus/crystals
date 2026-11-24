"""Stage 2 of the external benchmarks (training venv): GRACE-featurised pickles -> PyG views.

    python displacement/bench/build_views.py <in.pkl> <name> [--maxn 50]
writes displacement/bench_data/views/<name>.pt

Same fields as views.make_view, so train_disp / models / ext.wrapper work unchanged, plus
  split      0/1/2 = the authors' train/val/test
  mode       'fixed' (cell unchanged, DefiNet defect sets) or 'cell' (cell relaxes, E3Relax sets)
  cell_T     (1,3,3) relaxed cell = cell0 @ (I + T)  (rows = lattice vectors), 'cell' mode target
  cell_r     (1,3,3) relaxed cell; stress0 (1,3,3) GRACE stress of the unrelaxed structure (eV/A^3)
Graph: the authors' rule, radius 6 A with at most 50 nearest neighbours per atom (r6n50).
Targets: y = non-affine displacement, (s_r - s_u wrapped to [-0.5, 0.5)) @ cell0 (A); for 'fixed' this is
the minimum-image Cartesian displacement. loss_mask = real atoms (DefiNet vacancy dummies excluded).
Node features x ("our inputs"): GRACE per-atom energy, |F0|, defect one-hot (0/1/2), distance to the
nearest defect site (capped 15 A; defect sets only). Edge features: distance, same element, defect edge,
F0_i.u_ij, F0_j.u_ij.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data

HERE = os.path.dirname(os.path.abspath(__file__))


def wrap(x):
    return x - np.round(x)


def convert(r, maxn):
    z = r["z"].astype(np.int64); n = len(z)
    cu, cr = r["cell_u"], r["cell_r"]
    su = r["pos_u"] @ np.linalg.inv(cu); sr = r["pos_r"] @ np.linalg.inv(cr)
    y = wrap(sr - su) @ cu
    fixed = np.allclose(cu, cr, atol=1e-4)
    T = np.linalg.inv(cu) @ cr - np.eye(3)
    # edges: ase i = centre, j = neighbour, vec = pos[j] + S @ cell - pos[i]; keep maxn nearest per centre
    i, j, S, d = r["ei"].astype(np.int64), r["ej"].astype(np.int64), r["eS"].astype(np.float64), r["ed"].astype(np.float64)
    order = np.lexsort((d, i)); i, j, S, d = i[order], j[order], S[order], d[order]
    start = np.searchsorted(i, i, side="left"); rank = np.arange(len(i)) - start
    keep = rank < maxn
    i, j, S, d = i[keep], j[keep], S[keep], d[keep]
    vec = r["pos_u"][j] + S @ cu - r["pos_u"][i]
    u = vec / d[:, None]
    F = r["force0"].astype(np.float64)
    defect = r["defect"] if r["defect"] is not None else np.zeros(n, np.int8)
    defect = defect.astype(np.int64)
    real = np.arange(n) < r["n_real"]
    cols = [r["pe0"][:, None], np.linalg.norm(F, axis=1)[:, None]]
    if r["defect"] is not None:
        oh = np.eye(3)[defect]
        dsites = np.nonzero(defect > 0)[0]
        if len(dsites):
            sd = wrap(su[:, None, :] - su[None, dsites, :]) @ cu
            dist = np.linalg.norm(sd, axis=-1).min(1)
        else:
            dist = np.full(n, 15.0)
        cols += [oh, np.minimum(dist, 15.0)[:, None]]
    else:
        dist = np.zeros(n)
    x = np.concatenate(cols, 1)
    ea = np.stack([d, (z[i] == z[j]).astype(float), ((defect[i] > 0) | (defect[j] > 0)).astype(float),
                   (F[i] * u).sum(1), (F[j] * u).sum(1)], 1)
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)  # noqa: E731
    g = Data(x=t(x), z=t(z, torch.long), edge_index=t(np.stack([j, i]), torch.long), edge_attr=t(ea), edge_u=t(u),
             force0=t(F), y=t(np.where(real[:, None], y, 0.0)), loss_mask=t(real, torch.bool), num_nodes=n)
    g.dist = t(dist)
    g.pos0 = t(r["pos_u"]); g.cell0 = t(cu).view(1, 3, 3); g.cell_r = t(cr).view(1, 3, 3)
    g.cell_off = t(S); g.n_edges = torch.tensor([len(i)]); g.is_def = t(defect, torch.long)
    g.cell_T = t(T).view(1, 3, 3); g.stress0 = t(r["stress0"]).view(1, 3, 3)
    g.unstable = torch.tensor([False])
    g.null = torch.tensor([bool(np.abs(y[real]).max() < 1e-4) and (fixed or np.abs(T).max() < 1e-5)])
    g.cell_y = torch.zeros(1, 3)
    g.split = torch.tensor([{"train": 0, "val": 1, "test": 2}[r["split"]]])
    g.mode = "fixed" if fixed else "cell"
    g.folder = r["id"]; g.key = r.get("mat", ""); g.element_a = r.get("mat", "") or "x"; g.element_b = ""
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("name"); ap.add_argument("--maxn", type=int, default=50)
    a = ap.parse_args()
    recs = pickle.load(open(a.inp, "rb"))
    # The published DefiNet high-density val.csv is a copy of train.csv (test is disjoint). Drop the
    # duplicated "val" records and hold out every 8th training structure per material as validation
    # instead (350/50/50 per material = the paper's 8:1:1), identical for every model.
    tr_ids = {r["id"] for r in recs if r["split"] == "train"}
    dup = [r for r in recs if r["split"] == "val" and r["id"] in tr_ids]
    if dup:
        recs = [r for r in recs if not (r["split"] == "val" and r["id"] in tr_ids)]
        k = {}
        for r in recs:
            if r["split"] == "train":
                k[r["mat"]] = k.get(r["mat"], -1) + 1
                if k[r["mat"]] % 8 == 7:
                    r["split"] = "val"
        print(f"val == train in the published split: dropped {len(dup)} duplicates, re-held-out every 8th train structure")
    gs = [convert(r, a.maxn) for r in recs]
    modes = {g.mode for g in gs}
    print(f"{a.name}: {len(gs)} graphs, modes {modes}, nodes/graph {np.mean([g.num_nodes for g in gs]):.0f}, "
          f"edges/graph {np.mean([g.edge_index.shape[1] for g in gs]):.0f}, splits "
          f"{np.bincount([int(g.split) for g in gs], minlength=3).tolist()}, "
          f"mean|y| {np.mean([float(g.y[g.loss_mask].norm(dim=-1).mean()) for g in gs]):.4f} A, "
          f"max|y| {max(float(g.y.norm(dim=-1).max()) for g in gs):.2f} A, "
          f"|centroid shift| {np.mean([float(g.y[g.loss_mask].mean(0).norm()) for g in gs]):.4f} A")
    if len(modes) > 1:  # one mode per dataset: a cell that changes in any structure -> 'cell' everywhere
        for g in gs:
            g.mode = "cell"
    out = os.path.join(os.path.dirname(HERE), "bench_data", "views", f"{a.name}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(gs, out)
    print("->", out)


if __name__ == "__main__":
    main()
