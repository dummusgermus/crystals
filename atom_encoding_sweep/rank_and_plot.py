"""Rank atom-type encodings vs. the current type 1/2 encoding and plot the best ones.

Ranking: for every (split, domain, model) setting, paired difference to type12
(same split + seed) of test R_tot median (primary goal: net defect energy) and
test MAE, at the production checkpoint (best val R_tot). An encoding is kept
only if it beats type12 on average over all 8 settings in BOTH metrics
and the R_tot improvement is significant (one-sided Wilcoxon p < 0.05).

Outputs: atom_encoding_sweep/ranking.json, ranking.md, best_encodings_vs_current.png
"""

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from summarize import load

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = [(d, m) for d in ("point", "planar") for m in ("cgcnn", "transformer")]
SETTING_LABEL = {
    ("point", "cgcnn"): "Point · CGCNN",
    ("point", "transformer"): "Point · Transformer",
    ("planar", "cgcnn"): "Planar · CGCNN",
    ("planar", "transformer"): "Planar · Transformer",
}
SPLITS = ("random", "chem")
SPLIT_LABEL = {"random": "Random split (known compounds)", "chem": "Held-out compounds (unseen chemistry)"}
NICE = {
    "type12": "current: type 1/2",
    "none": "no type",
    "role_onehot": "role one-hot",
    "elem_onehot": "element one-hot",
    "elem_props": "element descriptors",
    "elem_props_role": "descriptors + role",
    "pair_props": "own + partner descriptors",
    "pair_props_edge": "own + partner descr. + edge feats",
    "onehot_props": "one-hot + descriptors + role",
}


def paired(runs, split, dom, mod, enc, metric):
    a, b = runs.get((split, dom, mod, enc), {}), runs.get((split, dom, mod, "type12"), {})
    common = sorted(set(a) & set(b))
    return np.array([a[s][metric] - b[s][metric] for s in common])


def rank(runs):
    encs = sorted({k[3] for k in runs} - {"type12"})
    table = []
    for enc in encs:
        row = {"encoding": enc, "settings": {}}
        d_r_all, d_m_rel_all, n_better = [], [], 0
        for split in SPLITS:
            for dom, mod in SETTINGS:
                dr = paired(runs, split, dom, mod, enc, "rtot")
                dm = paired(runs, split, dom, mod, enc, "mae")
                if not len(dr):
                    continue
                base_mae = np.mean([v["mae"] for v in runs[(split, dom, mod, "type12")].values()])
                key = f"{split}/{dom}/{mod}"
                row["settings"][key] = {
                    "d_rtot_pp": float(dr.mean()),
                    "d_mae_eV": float(dm.mean()),
                    "seed_wins_rtot": f"{int((dr < 0).sum())}/{len(dr)}",
                    "n_seeds": int(len(dr)),
                }
                d_r_all.extend(dr.tolist())
                d_m_rel_all.extend((100 * dm / base_mae).tolist())
                n_better += int(dr.mean() < 0)
        n_set = len(row["settings"])
        row["n_settings"] = n_set
        row["settings_improved_rtot"] = n_better
        row["mean_d_rtot_pp"] = float(np.mean([s["d_rtot_pp"] for s in row["settings"].values()]))
        row["mean_d_mae_rel_pct"] = float(np.mean(d_m_rel_all))
        # paired test over all (setting, seed) pairs
        row["p_rtot_wilcoxon"] = float(stats.wilcoxon(d_r_all, alternative="less").pvalue)
        row["p_mae_wilcoxon"] = float(stats.wilcoxon(d_m_rel_all, alternative="less").pvalue)
        row["better_than_current"] = bool(row["mean_d_rtot_pp"] < 0 and row["mean_d_mae_rel_pct"] < 0 and n_set == 8 and row["p_rtot_wilcoxon"] < 0.05)
        table.append(row)
    table.sort(key=lambda r: r["mean_d_rtot_pp"])
    return table


def plot(runs, best, path):
    ink, ink2, grid, axis = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    styles = {"type12": dict(color="#52514e", marker="D", ms=7, mfc="white", mew=2)}
    for enc, col, mk in zip(best, ["#2a78d6", "#eb6834", "#1baf7a"], ["o", "s", "^"]):
        styles[enc] = dict(color=col, marker=mk, ms=8, mfc=col, mew=0)
    encs = ["type12"] + best
    metrics = [("rtot", "Test net-energy error  R_tot median (%)"), ("mae", "Test per-atom MAE (meV/atom)")]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": axis, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), facecolor="#fcfcfb")
    off = np.linspace(-0.27, 0.27, len(encs))
    for i, split in enumerate(SPLITS):
        for j, (metric, xlabel) in enumerate(metrics):
            ax = axes[i, j]
            ax.set_facecolor("#fcfcfb")
            scale = 1000 if metric == "mae" else 1
            for k, enc in enumerate(encs):
                for y, (dom, mod) in enumerate(SETTINGS):
                    seeds = runs.get((split, dom, mod, enc), {})
                    if not seeds:
                        continue
                    v = np.array([s[metric] for s in seeds.values()]) * scale
                    yy = len(SETTINGS) - 1 - y + off[::-1][k]
                    st = styles[enc]
                    ax.errorbar(v.mean(), yy, xerr=v.std(), fmt="none", ecolor=st["color"],
                                elinewidth=1.5, capsize=0, alpha=0.55, zorder=2)
                    ax.plot(v.mean(), yy, linestyle="none", zorder=3,
                            label=NICE[enc] if (y == 0 and i == 0 and j == 0) else None,
                            markeredgecolor=st["color"], **st)
            ax.set_yticks(range(len(SETTINGS)))
            ax.set_yticklabels([SETTING_LABEL[s] for s in SETTINGS[::-1]])
            for y in range(len(SETTINGS) - 1):
                ax.axhline(y + 0.5, color=grid, lw=0.8, zorder=0)
            ax.grid(axis="x", color=grid, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(left=0)
            ax.set_ylim(-0.5, len(SETTINGS) - 0.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xlabel(xlabel + "   ← better", color=ink2)
            if j == 0:
                ax.set_title(SPLIT_LABEL[split], loc="left", color=ink, fontsize=11, fontweight="bold")
    n_r = max(len(v) for k, v in runs.items() if k[0] == "random")
    n_c = max(len(v) for k, v in runs.items() if k[0] == "chem")
    fig.suptitle("Atom-type encoding: best new encodings vs. current type 1/2 label",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=ink)
    fig.text(0.01, 0.948, f"Mean ± std over {n_r} (random) / {n_c} (held-out) paired seeds · "
             "production global-v2 setup, checkpoint = best val R_tot · lower is better",
             color=ink2, fontsize=10)
    fig.legend(loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=4, frameon=False, fontsize=10,
               handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.875))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())


def write_md(table, runs, path):
    lines = ["# Atom-type encoding ranking (vs. current type 1/2)", "",
             "Mean paired change vs. type12 over 8 settings (2 splits × point/planar × CGCNN/Transformer). "
             "Negative = better. p = one-sided Wilcoxon over all (setting, seed) pairs.", "",
             "| rank | encoding | ΔR_tot (pp) | ΔMAE (%) | settings improved (R_tot) | p(R_tot) | p(MAE) | better than current |",
             "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(table, 1):
        lines.append(f"| {i} | {r['encoding']} ({NICE[r['encoding']]}) | {r['mean_d_rtot_pp']:+.1f} | "
                     f"{r['mean_d_mae_rel_pct']:+.1f} | {r['settings_improved_rtot']}/{r['n_settings']} | "
                     f"{r['p_rtot_wilcoxon']:.3g} | {r['p_mae_wilcoxon']:.3g} | {'yes' if r['better_than_current'] else 'no'} |")
    lines += ["", "## Best encoding per setting (test R_tot median, %)", "",
              "| split | setting | current | best | encoding | seed wins |", "|---|---|---|---|---|---|"]
    for split in SPLITS:
        for dom, mod in SETTINGS:
            base = np.mean([v["rtot"] for v in runs[(split, dom, mod, "type12")].values()])
            cands = [(np.mean([v["rtot"] for v in s.values()]), k[3]) for k, s in runs.items()
                     if k[:3] == (split, dom, mod) and k[3] != "type12"]
            val, enc = min(cands)
            dr = paired(runs, split, dom, mod, enc, "rtot")
            lines.append(f"| {split} | {SETTING_LABEL[(dom, mod)]} | {base:.1f} | {val:.1f} | {enc} | {int((dr<0).sum())}/{len(dr)} |")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    runs = load()
    table = rank(runs)
    json.dump(table, open(os.path.join(HERE, "ranking.json"), "w"), indent=1)
    write_md(table, runs, os.path.join(HERE, "ranking.md"))
    good = [r for r in table if r["better_than_current"]]
    # plot: best by R_tot, best by MAE, then next-best by R_tot (top ones are statistically tied)
    best = [good[0]["encoding"]] if good else []
    for r in sorted(good, key=lambda r: r["mean_d_mae_rel_pct"])[:1] + good:
        if r["encoding"] not in best and len(best) < 3:
            best.append(r["encoding"])
    plot(runs, best, os.path.join(HERE, "best_encodings_vs_current.png"))
    print(open(os.path.join(HERE, "ranking.md")).read())
    print("plotted:", best)


if __name__ == "__main__":
    main()
