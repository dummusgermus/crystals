"""Old production checkpoints vs. retrained ones (mean/min/max over seeds) on the delivery val set.
Writes production_retrain/comparison.{md,json}."""
import glob, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["point_cgcnn", "planar_cgcnn", "point_transformer", "planar_transformer"]
METRICS = [("val_mae", "val MAE (meV/atom)", 1000), ("val_r_tot_graph_median", "R_tot graph-sum (%)", 1),
           ("val_r_tot_full_median", "R_tot full-cell (%)", 1), ("val_abs_total_err_graph_eV", "|dE| graph-sum (meV)", 1000)]
res = defaultdict(list)
for p in sorted(glob.glob(os.path.join(HERE, "results", "*.json"))):
    r = json.load(open(p)); res[r["model_key"]].append(r)
out, md = {}, ["# Production retrain: old vs. new (delivery validation set)\n",
               "New = mean over seeds [min - max]. Val set was also used for checkpoint selection (both old and new).\n"]
for k in KEYS:
    rs = res.get(k, [])
    if not rs:
        continue
    md += [f"\n## {k} ({len(rs)} seeds; target mode old={rs[0]['target_mode_old']}, new={rs[0]['target_mode_new']})\n",
           "| metric | old | new mean [min - max] | change |", "|---|---|---|---|"]
    out[k] = {}
    for m, lab, f in METRICS:
        old = rs[0]["old"][m] * f; new = np.array([r["new"][m] for r in rs]) * f
        out[k][m] = {"old": old, "new_mean": float(new.mean()), "new": new.tolist()}
        md.append(f"| {lab} | {old:.2f} | {new.mean():.2f} [{new.min():.2f} - {new.max():.2f}] | {100 * (new.mean() - old) / old:+.1f} % |")
json.dump(out, open(os.path.join(HERE, "comparison.json"), "w"), indent=1)
open(os.path.join(HERE, "comparison.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
