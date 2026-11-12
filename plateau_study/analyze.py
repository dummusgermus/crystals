"""Plateau study analysis -> plateau_study/analysis.json/.md

Per variant x split: stuck rate (best val MAE never below ESC meV), escape epoch
(first epoch with val MAE < ESC), final-epoch val MAE, and test metrics at three
checkpoint rules (best val R_tot / gated / best val MAE), paired vs. 'base'.
"""
import glob, json, os
from collections import defaultdict
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ESC = 0.020  # eV/atom: plateau sits at ~0.034, trained runs reach ~0.011-0.013
CK = ("ckpt_rtot", "ckpt_gated", "ckpt_mae")

def load(d=os.path.join(HERE, "results")):
    runs = defaultdict(dict)
    for p in glob.glob(os.path.join(d, "*.json")):
        r = json.load(open(p))
        vm = np.array(r["curves"]["val_mae"])
        below = np.where(vm < ESC)[0]
        e = {"stuck": bool(vm.min() >= ESC), "escape": int(below[0]) + 1 if len(below) else None,
             "final_val_mae": float(vm[-1]), "min_val_mae": float(vm.min()),
             "late_relapse": bool(len(below) and (vm[-200:] >= ESC).mean() > 0.5),
             "gn_p99": float(np.percentile(r["curves"]["grad_norm"], 99))}
        for c in CK:
            e[c] = {"mae": r[c]["test"]["mae"], "rtot": r[c]["test"]["r_tot_median"], "epoch": r[c]["epoch"]}
        runs[(r["variant"], r["split"])][r["seed"]] = e
    return runs

def main():
    runs = load()
    variants = sorted({v for v, _ in runs}, key=lambda v: (v != "base", v))
    out, lines = {}, ["# Plateau study: planar CGCNN, pair_props_edge, 2000 epochs", "",
        f"Stuck = val MAE never below {1000*ESC:.0f} meV/atom. Δ = paired vs. base (same split+seed), "
        "mean over random+chem; p = one-sided Wilcoxon (variant better).", ""]
    hdr = ("| variant | stuck | late relapse | median escape ep | "
           + " | ".join(f"{c[5:]}: R_tot / MAE" for c in CK)
           + " | ΔR_tot gated (pp) | ΔMAE gated (%) | p(R_tot) | p(MAE) |")
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for v in variants:
        allr = [e for sp in ("random", "chem") for e in runs.get((v, sp), {}).values()]
        row = {"n": len(allr), "stuck": sum(e["stuck"] for e in allr),
               "relapse": sum(e["late_relapse"] for e in allr),
               "escape_med": float(np.median([e["escape"] for e in allr if e["escape"]])) if any(e["escape"] for e in allr) else None}
        for c in CK:
            row[c] = {"rtot": float(np.mean([e[c]["rtot"] for e in allr])),
                      "mae": 1000 * float(np.mean([e[c]["mae"] for e in allr]))}
        row["per_split"] = {}
        dr, dm = [], []
        for sp in ("random", "chem"):
            a, b = runs.get((v, sp), {}), runs.get(("base", sp), {})
            com = sorted(set(a) & set(b))
            bm = np.mean([b[k]["ckpt_rtot"]["mae"] for k in com]) if com else 1
            for c in CK:
                row["per_split"].setdefault(sp, {})[c] = {
                    "rtot": float(np.mean([a[k][c]["rtot"] for k in a])),
                    "mae": 1000 * float(np.mean([a[k][c]["mae"] for k in a]))}
            row["per_split"][sp]["stuck"] = f"{sum(e['stuck'] for e in a.values())}/{len(a)}"
            # compare gated variant vs. CURRENT production (base @ ckpt_rtot)
            dr += [a[k]["ckpt_gated"]["rtot"] - b[k]["ckpt_rtot"]["rtot"] for k in com]
            dm += [100 * (a[k]["ckpt_gated"]["mae"] - b[k]["ckpt_rtot"]["mae"]) / bm for k in com]
        row["d_rtot_pp"], row["d_mae_pct"] = float(np.mean(dr)), float(np.mean(dm))
        row["p_rtot"] = float(stats.wilcoxon(dr, alternative="less").pvalue) if np.any(dr) else float("nan")
        row["p_mae"] = float(stats.wilcoxon(dm, alternative="less").pvalue) if np.any(dm) else float("nan")
        out[v] = row
        ck = " | ".join(f"{row[c]['rtot']:.1f} / {row[c]['mae']:.1f}" for c in CK)
        lines.append(f"| {v} | {row['stuck']}/{row['n']} | {row['relapse']}/{row['n']} | {row['escape_med']} | {ck} | "
                     f"{row['d_rtot_pp']:+.1f} | {row['d_mae_pct']:+.1f} | {row['p_rtot']:.3g} | {row['p_mae']:.3g} |")
    lines += ["", "Δ columns compare variant @ gated checkpoint vs. the CURRENT production setup (base @ best-val-R_tot).", "",
              "## Per split (test R_tot % / MAE meV at gated checkpoint, stuck count)", "",
              "| variant | random | chem |", "|---|---|---|"]
    for v in variants:
        ps = out[v]["per_split"]
        lines.append(f"| {v} | " + " | ".join(
            f"{ps[s]['ckpt_gated']['rtot']:.1f} / {ps[s]['ckpt_gated']['mae']:.1f} (stuck {ps[s]['stuck']})" for s in ("random", "chem")) + " |")
    json.dump(out, open(os.path.join(HERE, "analysis.json"), "w"), indent=1)
    open(os.path.join(HERE, "analysis.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
