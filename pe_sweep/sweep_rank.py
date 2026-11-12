"""Generic paired ranking of sweep variants vs. a baseline variant.

    python pe_sweep/sweep_rank.py --dir pe_sweep --field pe --baseline none
    python pe_sweep/sweep_rank.py --dir atom_encoding_sweep_long --field encoding --baseline type12

For every (split, domain, model) setting: paired test deltas vs. the baseline
(same split + seed) at the production checkpoint (best val R_tot). Ranks overall
and per model kind (Transformer-only variants are only compared on Transformer
settings). Writes <dir>/ranking_<field>.json and .md.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

SETTINGS = [(s, d, m) for s in ("random", "chem") for d in ("point", "planar")
            for m in ("cgcnn", "transformer")]


def load(directory, field):
    runs = defaultdict(dict)
    for path in glob.glob(os.path.join(directory, "results", "*.json")):
        r = json.load(open(path))
        t = r["ckpt_rtot"]["test"]
        runs[(r["split"], r["domain"], r["model_kind"], r.get(field, "none"))][r["seed"]] = {
            "mae": t["mae"], "rtot": t["r_tot_median"], "dE": t["abs_total_err_eV"],
            "mae_bestmae": r["ckpt_mae"]["test"]["mae"],
        }
    return runs


def mean(runs, key, metric):
    v = runs.get(key, {})
    return float(np.mean([s[metric] for s in v.values()])) if v else float("nan")


def rank(runs, baseline, models=("cgcnn", "transformer")):
    variants = sorted({k[3] for k in runs if k[2] in models} - {baseline})
    table = []
    for var in variants:
        dr_all, dm_all, per = [], [], {}
        for s, d, m in SETTINGS:
            if m not in models:
                continue
            a, b = runs.get((s, d, m, var), {}), runs.get((s, d, m, baseline), {})
            common = sorted(set(a) & set(b))
            if not common:
                continue
            dr = np.array([a[k]["rtot"] - b[k]["rtot"] for k in common])
            base_mae = np.mean([b[k]["mae"] for k in common])
            dm = np.array([100 * (a[k]["mae"] - b[k]["mae"]) / base_mae for k in common])
            per[f"{s}/{d}/{m}"] = {"d_rtot_pp": float(dr.mean()), "d_mae_pct": float(dm.mean()),
                                    "wins": f"{int((dr < 0).sum())}/{len(dr)}"}
            dr_all += dr.tolist()
            dm_all += dm.tolist()
        if not per:
            continue
        pr = stats.wilcoxon(dr_all, alternative="less").pvalue if len(dr_all) > 5 else float("nan")
        pm = stats.wilcoxon(dm_all, alternative="less").pvalue if len(dm_all) > 5 else float("nan")
        table.append({
            "variant": var, "n_settings": len(per), "n_pairs": len(dr_all),
            "mean_d_rtot_pp": float(np.mean([v["d_rtot_pp"] for v in per.values()])),
            "mean_d_mae_pct": float(np.mean([v["d_mae_pct"] for v in per.values()])),
            "settings_improved_rtot": sum(v["d_rtot_pp"] < 0 for v in per.values()),
            "settings_improved_mae": sum(v["d_mae_pct"] < 0 for v in per.values()),
            "p_rtot": float(pr), "p_mae": float(pm), "settings": per,
        })
    table.sort(key=lambda r: r["mean_d_rtot_pp"])
    return table


def md_table(title, table):
    lines = [f"### {title}", "",
             "| rank | variant | ΔR_tot (pp) | ΔMAE (%) | settings better R_tot | settings better MAE | p(R_tot) | p(MAE) |",
             "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(table, 1):
        lines.append(f"| {i} | {r['variant']} | {r['mean_d_rtot_pp']:+.2f} | {r['mean_d_mae_pct']:+.1f} | "
                     f"{r['settings_improved_rtot']}/{r['n_settings']} | {r['settings_improved_mae']}/{r['n_settings']} | "
                     f"{r['p_rtot']:.3g} | {r['p_mae']:.3g} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--baseline", required=True)
    a = ap.parse_args()
    runs = load(a.dir, a.field)
    out = {"all": rank(runs, a.baseline), "cgcnn": rank(runs, a.baseline, ("cgcnn",)),
           "transformer": rank(runs, a.baseline, ("transformer",))}
    lines = [f"# Ranking of `{a.field}` vs. `{a.baseline}` ({sum(len(v) for v in runs.values())} runs)", "",
             "Paired test deltas (same split + seed), checkpoint = best val R_tot. Negative = better. "
             "p = one-sided Wilcoxon over all (setting, seed) pairs.", ""]
    for k, title in (("all", "All settings"), ("cgcnn", "CGCNN only"), ("transformer", "Transformer only")):
        lines += md_table(title, out[k]) + [""]
    lines += ["### Mean test R_tot (%) / MAE (meV/atom) per setting", "",
              "| setting | " + " | ".join(sorted({k[3] for k in runs})) + " |",
              "|---|" + "---|" * len({k[3] for k in runs})]
    for s, d, m in SETTINGS:
        cells = [f"{mean(runs, (s, d, m, v), 'rtot'):.1f} / {1000 * mean(runs, (s, d, m, v), 'mae'):.1f}"
                 for v in sorted({k[3] for k in runs})]
        lines.append(f"| {s}/{d}/{m} | " + " | ".join(cells) + " |")
    json.dump(out, open(os.path.join(a.dir, f"ranking_{a.field}.json"), "w"), indent=1)
    open(os.path.join(a.dir, f"ranking_{a.field}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
