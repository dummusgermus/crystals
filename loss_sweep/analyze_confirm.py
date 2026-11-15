"""Loss confirmation: 9 seeds (0-8) per candidate vs base, per (model, domain, split).
Writes loss_sweep/analysis_confirm.{md,json}."""
import glob, json, os
from collections import defaultdict
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = {("cgcnn", "point"): ["flip-target"], ("transformer", "point"): ["lamx3"],
        ("transformer", "planar"): ["lamx10", "lamx30"]}
runs = defaultdict(dict)
for p in glob.glob(os.path.join(HERE, "results", "*.json")):
    r = json.load(open(p)); t = r["ckpt_rtot"]["test"]
    runs[(r["model_kind"], r["domain"], r["split"], r["variant"])][r["seed"]] = (t["r_tot_median"], t["mae"], t["abs_total_err_eV"])
out, md = {}, ["# Loss confirmation (seeds 0-8, paired vs base, checkpoint = best val R_tot)\n",
               "Negative = better. p = two-sided Wilcoxon signed-rank over seeds.\n",
               "| model | domain | split | variant | n | base R_tot % | var R_tot % | ΔR_tot pp | wins | p | base MAE | var MAE | ΔMAE % | p(MAE) | Δ|ΔE| % |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for (m, d), vs in CAND.items():
    for v in vs:
        pooled_r, pooled_m = [], []
        for s in ("random", "chem"):
            a, b = runs[(m, d, s, v)], runs[(m, d, s, "base")]
            c = sorted(set(a) & set(b))
            A, B = np.array([a[k] for k in c]), np.array([b[k] for k in c])
            dr = A[:, 0] - B[:, 0]; dm = 100 * (A[:, 1] - B[:, 1]) / B[:, 1].mean(); de = 100 * (A[:, 2] - B[:, 2]) / B[:, 2].mean()
            pr = stats.wilcoxon(dr).pvalue; pm = stats.wilcoxon(dm).pvalue
            row = dict(n=len(c), base_rtot=B[:, 0].mean(), var_rtot=A[:, 0].mean(), d_rtot=dr.mean(), wins=int((dr < 0).sum()),
                       p_rtot=pr, base_mae=B[:, 1].mean(), var_mae=A[:, 1].mean(), d_mae=dm.mean(), p_mae=pm, d_dE=de.mean(),
                       base_mae_med=float(np.median(B[:, 1])), var_mae_med=float(np.median(A[:, 1])))
            out[f"{m}/{d}/{s}/{v}"] = {k: float(x) for k, x in row.items()}
            pooled_r += dr.tolist(); pooled_m += dm.tolist()
            md.append(f"| {m} | {d} | {s} | {v} | {len(c)} | {row['base_rtot']:.1f} | {row['var_rtot']:.1f} | {dr.mean():+.2f} | {row['wins']}/{len(c)} | {pr:.3g} | {row['base_mae']:.2f} | {row['var_mae']:.2f} | {dm.mean():+.1f} | {pm:.3g} | {de.mean():+.1f} |")
        out[f"{m}/{d}/pooled/{v}"] = dict(d_rtot=float(np.mean(pooled_r)), p_rtot=float(stats.wilcoxon(pooled_r).pvalue),
                                         d_mae=float(np.mean(pooled_m)), p_mae=float(stats.wilcoxon(pooled_m).pvalue))
        o = out[f"{m}/{d}/pooled/{v}"]
        md.append(f"| {m} | {d} | **pooled** | {v} | {len(pooled_r)} | | | {o['d_rtot']:+.2f} | | {o['p_rtot']:.3g} | | | {o['d_mae']:+.1f} | {o['p_mae']:.3g} | |")
json.dump(out, open(os.path.join(HERE, "analysis_confirm.json"), "w"), indent=1)
open(os.path.join(HERE, "analysis_confirm.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
