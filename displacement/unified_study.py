"""Unified point+planar+mixed study summary (runs_unified, job 5482173).
    python displacement/unified_study.py   -> prints table, writes runs_unified/unified_study.json
Per config: mean +/- std over the 6 runs (3 splits x 2 seeds) of the single-run test metrics in results/*.json:
  r_disp point (8.9 A region) / planar / mixed (all atoms, antisite region, pure-fault cells), mixed c-vector MAE,
  energy relative error (energy head; with --e-harm head + harmonic term) per domain.
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "runs_unified", "results")
COLS = [("point_r_disp", "r pt"), ("planar_r_disp", "r pl"), ("mixed_r_disp_all", "r mx"),
        ("mixed_pd_r_disp", "r mx-pd"), ("mixed_none_r_disp", "r mx-none"), ("mixed_cell_mae", "c mx A"),
        ("point_e_rel", "E pt"), ("planar_e_rel", "E pl"), ("mixed_e_rel", "E mx")]


def label(key):
    m = "PaiNN" if "_painn_" in key else "DefiNet"
    tr = "mixed-trained" if "_mx" in key else "zero-shot"
    e = ("ew0.3 harm-res detached" if "_ehd" in key else "ew0.3 harm-res" if "_eh" in key
         else "ew0.3" if "_ew0.3" in key else "pos-only")
    return f"{m} {tr} {e}"


def main():
    groups = {}
    for f in sorted(glob.glob(f"{RD}/*.json")):
        key = re.sub(r"_s\d+_sp\d+\.json$", "", os.path.basename(f))
        groups.setdefault(key, []).append(json.load(open(f))["test"])
    res = {}
    print(f"{'config':40s} n  " + " ".join(f"{c:>12s}" for _, c in COLS))
    for key, js in sorted(groups.items(), key=lambda kv: label(kv[0])):
        r = {"key": key, "n_runs": len(js)}
        for k, _ in COLS:
            v = np.array([j.get(k, np.nan) for j in js], dtype=float)
            r[k], r[k + "_std"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan, float(np.nanstd(v)) if np.isfinite(v).any() else np.nan
        res[label(key)] = r
        print(f"{label(key):40s} {len(js)}  " + " ".join(f"{r[k]:6.3f}+-{r[k + '_std']:.3f}"[:12].rjust(12) for k, _ in COLS))
    json.dump(res, open(os.path.join(HERE, "runs_unified", "unified_study.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
