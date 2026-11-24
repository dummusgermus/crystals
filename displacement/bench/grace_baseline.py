"""GRACE-relaxation baseline on the benchmark test sets: relax the unrelaxed test structure with GRACE
(grace_feats.py --relax-test: FIRE, fmax 0.02 eV/A, <= 500 steps, FrechetCellFilter when the cell moves)
and score it like a model prediction (same metrics as train_disp.bench_metrics).

    python displacement/bench/grace_baseline.py <prep.pkl> [...]
prints one row per dataset and writes displacement/bench_data/grace_baseline.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def score(r):
    n = r["n_real"]
    cr, c0 = r["cell_r"], r["cell_u"]
    xr, x0 = r["pos_r"][:n], r["pos_u"][:n]
    g = r["grace_relaxed"]
    xp, cp = g["pos"][:n], g["cell"]

    def mae(x):
        df = (x - xr) @ np.linalg.inv(cr)
        return float(np.abs((df - np.round(df)) @ cr).mean())

    G = lambda c: c @ c.T  # noqa: E731
    # r_disp on the same footing as our models: |non-affine displacement error| / |true displacement|
    su, sr, sp = x0 @ np.linalg.inv(c0), xr @ np.linalg.inv(cr), xp @ np.linalg.inv(cp)
    w = lambda s: s - np.round(s)  # noqa: E731
    yt, yp = w(sr - su) @ c0, w(sp - su) @ c0
    den = np.linalg.norm(yt)
    return dict(mae_pos=mae(xp), mae_pos_dummy=mae(x0), mae_metric=float(np.linalg.norm(G(cp) - G(cr))),
                mae_vol=float(abs(np.linalg.det(cp) - np.linalg.det(cr))),
                mae_metric_dummy=float(np.linalg.norm(G(c0) - G(cr))),
                mae_vol_dummy=float(abs(np.linalg.det(c0) - np.linalg.det(cr))),
                r_disp=float(np.linalg.norm(yp - yt) / den) if den > 1e-6 else np.nan,
                conv=float(g["conv"]), steps=g["steps"])


def main():
    out_p = os.path.join(os.path.dirname(HERE), "bench_data", "grace_baseline.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for p in sys.argv[1:]:
        name = os.path.basename(p)[:-4]
        recs = [r for r in pickle.load(open(p, "rb")) if r["split"] == "test" and r.get("grace_relaxed")]
        s = [score(r) for r in recs]
        agg = {k: float(np.nanmean([x[k] for x in s])) for k in s[0] if k != "r_disp"}
        agg["r_disp"] = float(np.nanmedian([x["r_disp"] for x in s]))
        agg["n"] = len(s)
        res[name] = agg
        print(f"{name:14s} n={len(s):5d} pos {agg['mae_pos']:.4f} (unrelaxed {agg['mae_pos_dummy']:.4f}) "
              f"metric {agg['mae_metric']:.3f} ({agg['mae_metric_dummy']:.3f}) vol {agg['mae_vol']:.3f} ({agg['mae_vol_dummy']:.3f}) r_disp {agg['r_disp']:.3f} "
              f"conv {agg['conv']:.2f} steps {agg['steps']:.0f}")
    json.dump(res, open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main()
