"""Inference runtime: one unified model (positions + energy) vs two separate models (position-only + energy-only).
    python displacement/bench_runtime.py [--device cuda|cpu] [--n 64]
Separate energy and position models have the same architecture as the unified one (PaiNN h512, 3 layers), so
"two models" = two forward passes of that architecture. Times per cell, on the mixed cells (162-384 atoms) and
on point-defect cells, for each stage:
  featurise  raw structure -> graph (radius edges, shell ranks, 3-/4-cycle counts; build_mixed.build)
  prepare    graph -> model input (view, unified channels, encodings, normalisation); shared by both models
  forward    one model forward (displacements + cell + per-atom energy)
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "mixed"))
from predict_disp import build, views  # noqa: E402
from train_disp import atom_energy, joint_gauge  # noqa: E402

PRE = "joint_painn_rt8.9_buf4_k3_hk3_base+cyc+force_st1_eqvF_l1_norm_std_ema_nodf_"


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def forward_all(model, vs, dev, bs, e_harm):
    for b in DataLoader(vs, batch_size=bs):
        b = b.to(dev)
        d, cell = model(b)
        dg = joint_gauge(model, b, d)
        if getattr(model, "e_atom", None) is not None:
            atom_energy(model, b, dg, e_harm, False)
    sync(dev)


def timeit(f, rep=3):
    f()  # warm-up
    ts = []
    for _ in range(rep):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda"); ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device(a.device); torch.set_num_threads(8)
    ck = torch.load(glob.glob(f"{HERE}/runs_unified/preds/{PRE}ew0.3_uni_mx_eh_dfrecipe_s0_sp0.pt")[0],
                    weights_only=False, map_location="cpu")
    model = build(ck, dev)
    print(f"device {a.device} | params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    # featurisation cost on raw mixed cells
    from build_mixed import build as featurise
    cells = pickle.load(open(f"{HERE}/mixed_data/cells.pkl", "rb"))[:16]
    rel = {}
    for p in sorted(glob.glob(f"{HERE}/mixed_data/relaxed_part*.pkl"))[:2]:
        rel.update({r["id"]: r for r in pickle.load(open(p, "rb"))})
    cells = [c for c in cells if c["id"] in rel][:8]
    t = time.perf_counter()
    for c in cells:
        featurise(c, rel[c["id"]])
    t_feat = (time.perf_counter() - t) / len(cells)
    sets = {"mixed": [g for g in torch.load(f"{HERE}/data/superset_mixed.pt", weights_only=False)][:a.n],
            "point": [g for g in torch.load(f"{HERE}/data/superset_point.pt", weights_only=False)][:a.n]}
    for name, gs in sets.items():
        t = time.perf_counter(); vs = views(gs, ck); t_prep = (time.perf_counter() - t) / len(gs)
        nat = np.mean([v.num_nodes for v in vs])
        for bs in (1, 16):
            t_fwd = timeit(lambda: forward_all(model, vs, dev, bs, True)) / len(gs)
            one = t_prep + t_fwd; two = t_prep + 2 * t_fwd
            print(f"{name:6s} ({nat:.0f} atoms/cell) bs {bs:2d} | per cell: prepare {1e3 * t_prep:6.1f} ms, forward "
                  f"{1e3 * t_fwd:6.1f} ms | unified {1e3 * one:6.1f} ms, two models {1e3 * two:6.1f} ms "
                  f"(+{100 * (two / one - 1):.0f}%)", flush=True)
    print(f"featurise raw mixed cell (edges + cycle counts, CPU): {1e3 * t_feat:.0f} ms/cell (shared by both setups)")


if __name__ == "__main__":
    main()
