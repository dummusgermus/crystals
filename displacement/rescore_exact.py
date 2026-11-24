"""Old (dump-sum) vs exact point energy labels, scored on the SAME labels and cells.
    python displacement/rescore_exact.py   -> runs_exact/exact_study.txt
Every run's saved test items (--save-preds) are rescored: point energy against dE_exact (data/superset_point_exact.pt,
full-cell LAMMPS totals) and against the old view sum of dump per-atom energies; all metrics on the cells that are
stable under the new flags (8 far-field cells excluded for both). Planar / mixed labels are unchanged.
Metrics: r_disp (point: 8.9 A region) and median relative energy error (|true| > 1 meV), mean +- std over runs.
"""
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from train_disp import R_EVAL_POINT, metrics  # noqa: E402

PRE = "joint_painn_rt8.9_buf4_k3_hk3_base+cyc+force_st1_eqvF_l1_norm_std_ema_nodf_ew0.3_uni_mx_eh_"
SETS = [("3 splits x 2 seeds, old labels", f"runs_unified/preds/{PRE}dfrecipe_s*_sp*.pt"),
        ("3 splits x 2 seeds, exact labels", f"runs_exact/preds/{PRE}ex_dfrecipe_s*_sp*.pt"),
        ("final (held-out 10%), old labels", f"runs_final_unified/preds/{PRE}final_dfrecipe_s*_sp0.pt"),
        ("final (held-out 10%), exact labels", f"runs_exact/preds/{PRE}ex_final_dfrecipe_s*_sp0.pt")]
EX = {(g.folder, g.defect_key): g for g in torch.load(os.path.join(HERE, "data", "superset_point_exact.pt"), weights_only=False)}


def erel(p, t):
    p, t = np.array(p), np.array(t); big = np.abs(t) > 1e-3
    return float(np.median(np.abs(p - t)[big] / np.abs(t[big])))


def score(f):
    items = torch.load(f, weights_only=False, map_location="cpu")["test"]
    r = {}
    for k, dom in enumerate(("point", "planar", "mixed")):
        its = [dict(it) for it in items if it["dom"] == k]
        if dom == "point":
            for it in its:
                it["unstable"] = bool(EX[(it["folder"], it["key"])].unstable)
        st = [it for it in its if not it["unstable"]]
        r[f"{dom} r_disp"] = metrics(its, R_EVAL_POINT if dom == "point" else None)["r_disp"]
        pred = [float(it["e_pred"].sum()) for it in st]
        if dom == "point":
            r["point E exact"] = erel(pred, [EX[(it["folder"], it["key"])].dE_exact for it in st])
            r["point E old"] = erel(pred, [float(it["e_true"].sum()) for it in st])
        else:
            r[f"{dom} E"] = erel(pred, [float(it["e_true"].sum()) for it in st])
    return r


def main():
    lines = []
    for name, pat in SETS:
        fs = sorted(glob.glob(os.path.join(HERE, pat)))
        if not fs:
            lines.append(f"{name}: no files"); continue
        rs = [score(f) for f in fs]
        lines.append(f"{name} ({len(fs)} runs)")
        lines.append("   " + " | ".join(f"{k} {np.mean([r[k] for r in rs]):.3f}+-{np.std([r[k] for r in rs]):.3f}" for k in rs[0]))
    open(os.path.join(HERE, "runs_exact", "exact_study.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
