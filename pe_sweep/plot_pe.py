"""Plot PE sweep: paired change vs. no PE, per variant and model (pe_sweep/pe_sweep_results.png)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NICE = {
    "rwse8": "RWSE 8", "rwse16": "RWSE 16", "lap8": "LapPE 8", "lap16": "LapPE 16",
    "lap8+rwse16": "LapPE 8 + RWSE 16", "spd": "hop-distance bias", "geo": "bond-path-distance bias",
    "spd+geo": "hop + bond-path bias", "rwse16+spd+geo": "RWSE 16 + hop + bond-path",
    "lap8+rwse16+spd+geo": "LapPE + RWSE + hop + bond-path",
}
ORDER = ["rwse8", "rwse16", "lap8", "lap16", "lap8+rwse16", "spd", "geo", "spd+geo",
         "rwse16+spd+geo", "lap8+rwse16+spd+geo"]


def main():
    rk = json.load(open(os.path.join(HERE, "ranking_pe.json")))
    ink, ink2, grid, axis = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    col = {"cgcnn": "#2a78d6", "transformer": "#eb6834"}
    mk = {"cgcnn": "o", "transformer": "s"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": axis, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), facecolor="#fcfcfb", sharey=True)
    for ax, (key, label) in zip(axes, [("mean_d_rtot_pp", "Change in test R_tot median (pp)"),
                                       ("mean_d_mae_pct", "Change in test per-atom MAE (%)")]):
        ax.set_facecolor("#fcfcfb")
        for k, model in enumerate(("cgcnn", "transformer")):
            rows = {r["variant"]: r for r in rk[model]}
            for y, v in enumerate(ORDER):
                if v not in rows:
                    continue
                r = rows[v]
                vals = [s["d_rtot_pp" if key == "mean_d_rtot_pp" else "d_mae_pct"] for s in r["settings"].values()]
                yy = len(ORDER) - 1 - y + (0.17 if model == "cgcnn" else -0.17)
                ax.plot(vals, [yy] * len(vals), linestyle="none", marker=mk[model], ms=4,
                        color=col[model], alpha=0.35, zorder=2)
                ax.plot(r[key], yy, linestyle="none", marker=mk[model], ms=9, color=col[model],
                        markeredgecolor="#fcfcfb", mew=1.5, zorder=3,
                        label=("CGCNN" if model == "cgcnn" else "Transformer") + " (mean of 4 settings)"
                        if (y == 0 or (model == "transformer" and v == "rwse8")) and ax is axes[0] else None)
        ax.axvline(0, color=ink2, lw=1.2, zorder=1)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([NICE[v] for v in ORDER[::-1]])
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.axhline(4.5, color=axis, lw=1, ls="--")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel(label + "   ← better than no PE", color=ink2)
    axes[0].text(axes[0].get_xlim()[1], 4.6, "Transformer-only attention biases ↓ ", ha="right",
                 va="bottom", color=ink2, fontsize=9)
    fig.suptitle("Positional encodings vs. none: no variant gives a reliable gain",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", color=ink)
    fig.text(0.01, 0.925, "204 runs · atom encoding pair_props_edge · paired with no-PE runs (same split + seed), "
             "3 seeds × point/planar × random/held-out · big marker = mean, small = per setting",
             color=ink2, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.905), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(os.path.join(HERE, "pe_sweep_results.png"), dpi=160, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
