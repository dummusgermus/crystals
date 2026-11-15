"""Confirmation of the plateau fix on point CGCNN and Transformer (planar+point).
Paired vs. 'base' (same split + seed). Variant at gated ckpt vs. base at ckpt_rtot (production),
plus both at ckpt_rtot. Writes plateau_study/analysis_confirm.md/json."""
import glob, json, os
from collections import defaultdict
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
runs = defaultdict(dict)
for p in glob.glob(os.path.join(HERE, "results", "*.json")):
    r = json.load(open(p))
    runs[(r["model_kind"], r["domain"], r["variant"], r["split"])][r["seed"]] = r
out, lines = {}, ["# Plateau fix confirmation", "",
    "Paired vs. current recipe (same split+seed). 'gated' = variant @ safer checkpoint vs. base @ best-val-R_tot (production). "
    "R_tot in %, MAE in meV/atom; Δ averaged over random+chem; wins = paired runs with lower test R_tot.", "",
    "| model / domain | variant | base R_tot / MAE | variant R_tot / MAE (gated) | ΔR_tot pp | ΔMAE % | wins R_tot | wins MAE | p(R_tot) | p(MAE) | ΔR_tot pp (both ckpt_rtot) |",
    "|---|---|---|---|---|---|---|---|---|---|---|"]
for model, dom in (("cgcnn", "planar"), ("cgcnn", "point"), ("transformer", "planar"), ("transformer", "point")):
    vs = sorted({k[2] for k in runs if k[0] == model and k[1] == dom} - {"base"})
    for v in vs:
        dr, dm, dr2, bR, bM, vR, vM = [], [], [], [], [], [], []
        for sp in ("random", "chem"):
            a, b = runs[(model, dom, v, sp)], runs[(model, dom, "base", sp)]
            for s in sorted(set(a) & set(b)):
                ta, tb = a[s]["ckpt_gated"]["test"], b[s]["ckpt_rtot"]["test"]
                dr.append(ta["r_tot_median"] - tb["r_tot_median"]); dm.append(100 * (ta["mae"] - tb["mae"]) / tb["mae"])
                dr2.append(a[s]["ckpt_rtot"]["test"]["r_tot_median"] - tb["r_tot_median"])
                bR.append(tb["r_tot_median"]); bM.append(1000 * tb["mae"]); vR.append(ta["r_tot_median"]); vM.append(1000 * ta["mae"])
        p1 = stats.wilcoxon(dr, alternative="less").pvalue; p2 = stats.wilcoxon(dm, alternative="less").pvalue
        row = dict(n=len(dr), d_rtot=float(np.mean(dr)), d_mae=float(np.mean(dm)), wins_r=int((np.array(dr) < 0).sum()),
                   wins_m=int((np.array(dm) < 0).sum()), p_rtot=float(p1), p_mae=float(p2), d_rtot_ckrtot=float(np.mean(dr2)),
                   base=(float(np.mean(bR)), float(np.mean(bM))), var=(float(np.mean(vR)), float(np.mean(vM))))
        out[f"{model}/{dom}/{v}"] = row
        lines.append(f"| {model} / {dom} | {v} | {row['base'][0]:.1f} / {row['base'][1]:.1f} | {row['var'][0]:.1f} / {row['var'][1]:.1f} | "
                     f"{row['d_rtot']:+.1f} | {row['d_mae']:+.1f} | {row['wins_r']}/{row['n']} | {row['wins_m']}/{row['n']} | "
                     f"{p1:.3g} | {p2:.3g} | {row['d_rtot_ckrtot']:+.1f} |")
json.dump(out, open(os.path.join(HERE, "analysis_confirm.json"), "w"), indent=1)
open(os.path.join(HERE, "analysis_confirm.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
