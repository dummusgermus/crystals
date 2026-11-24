"""Mixed-defect cells, stage 2 (training venv): cells.pkl + relaxed_part*.pkl -> data/superset_mixed.pt

Same graph assembly as the planar training data (build_superset.assemble: radius-6 edges with shell ranks,
3-/4-cycle counts, all atoms), domain 'mixed', plus the fault channels for the unified model:
  is_defect / dvec / dist   the point defect (antisite; zeros for kind 'none')
  is_fault / dist_fault     fault-layer atoms (planar is_defect labels) and the distance to the nearest one
Targets as planar: non-affine displacement with the mean removed, c-vector change (cell_change_frac),
per-atom pe0 / pe1 from GRACE. Unstable = planar rule (max |d| > 1 A or |dc_z| > 0.1).

    python displacement/mixed/build_mixed.py [--parts 'displacement/mixed_data/relaxed_part*.pkl']
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); DISP = os.path.dirname(HERE)
sys.path.insert(0, DISP)
from build_superset import SYMBOL_TO_Z, assemble, nearest_defect_vec  # noqa: E402


def build(c, r):
    pos, H0, T = c["pos"], c["cell"], c["types"]
    H1, p1 = r["cell1"], r["pos1"]
    s = p1 @ np.linalg.inv(H1) - pos @ np.linalg.inv(H0); s -= np.round(s)
    d = s @ H0; d -= d.mean(0)
    dc = (H1[2] - H0[2]) @ np.linalg.inv(H0)
    pi = c["point_idx"]
    is_def = np.zeros(len(pos))
    if pi >= 0:
        is_def[pi] = 1.0
    dvec = nearest_defect_vec(pos, H0, [pi] if pi >= 0 else [])
    dfault = np.linalg.norm(nearest_defect_vec(pos, H0, np.flatnonzero(c["is_fault"])), axis=1)
    A, B = c["elements"]
    md = float(np.linalg.norm(d, axis=1).max())
    meta = dict(domain="mixed", folder=c["folder"], element_a=A, element_b=B, defect_key=c["kind"],
                stack_sequence=c["folder"].split("_", 1)[1], kind=c["kind"], compound=c["compound"],
                split_test=list(c["split_test"]), dist_fault_of_point=c["dist_fault_of_point"],
                dE_total=float(r["pe1"].sum() - r["pe0"].sum()), n_full=len(pos),
                cell_change_frac=torch.tensor(dc, dtype=torch.float).unsqueeze(0), max_disp=md,
                unstable=bool(md > 1.0 or abs(dc[2]) > 0.1), conv=bool(r["conv"]), reps=tuple(c["reps"]))
    g = assemble(np.arange(len(pos)), pos, H0, T, r["pe0"], r["pe1"], r["force0"], d, dvec, is_def,
                 {1: SYMBOL_TO_Z[A], 2: SYMBOL_TO_Z[B]}, meta)
    g.is_fault = torch.tensor(c["is_fault"], dtype=torch.float)
    g.dist_fault = torch.tensor(dfault, dtype=torch.float)
    g.rel = torch.tensor(pos, dtype=torch.float)
    g.full_cell = True
    return g


def _one(args):
    return build(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(DISP, "mixed_data", "cells.pkl"))
    ap.add_argument("--parts", default=os.path.join(DISP, "mixed_data", "relaxed_part*.pkl"))
    ap.add_argument("--out", default=os.path.join(DISP, "data", "superset_mixed.pt"))
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    cells = {c["id"]: c for c in pickle.load(open(a.cells, "rb"))}
    rel = {}
    for p in sorted(glob.glob(a.parts)):
        for r in pickle.load(open(p, "rb")):
            rel[r["id"]] = r
    ids = [i for i in cells if i in rel]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        gs = list(ex.map(_one, [(cells[i], rel[i]) for i in ids], chunksize=4))
    torch.save(gs, a.out)
    k = np.array([g.kind for g in gs]); u = np.array([g.unstable for g in gs]); cv = np.array([g.conv for g in gs])
    print(f"{len(gs)}/{len(cells)} cells -> {a.out} | unstable {u.mean():.2f} | converged {cv.mean():.2f} | "
          + " ".join(f"{t}: dE med {np.median([g.dE_total for g in gs if g.kind == t]):.3f} eV" for t in ("none", "near", "far", "rand")))


if __name__ == "__main__":
    main()
