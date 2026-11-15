"""Ensemble size vs. test error change (ensemble_study/ensemble_results.png)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "analysis.json")))
ink, ink2, grid, axis, bg = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
col = {"cgcnn": "#2a78d6", "transformer": "#eb6834"}; ls = {"point": "-", "planar": "--"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), facecolor=bg)
for ax, (metric, lab) in zip(axes, [("rtot", "Change in test R_tot median (pp)"), ("mae", "Change in test per-atom MAE (%)")]):
    ax.set_facecolor(bg)
    for mdl in ("cgcnn", "transformer"):
        for dom in ("point", "planar"):
            rows = [v for k, v in r.items() if k.split("/")[1:3] == [dom, mdl]]
            def d(meth):
                out = []
                for v in rows:
                    key = ("rtot_full" if v["own_mode"] == "full" else "rtot_graph") if metric == "rtot" else "mae"
                    out.append(v[meth][key] - v["single"][key] if metric == "rtot" else 100 * (v[meth][key] - v["single"][key]) / v["single"][key])
                return np.mean(out)
            ys = [0.0] + [d(f"ens-{k}") for k in range(2, 6)]
            name = ("CGCNN" if mdl == "cgcnn" else "Transformer") + f" · {dom}"
            ax.plot(range(1, 6), ys, ls[dom], color=col[mdl], lw=2, marker="o", ms=6, mec=bg, mew=1.2, label=name)
            ax.plot(0.55, d("best-val"), marker="D", ms=7, mfc="none", mec=col[mdl], mew=1.6, ls="none")
    ax.axhline(0, color=ink2, lw=1)
    ax.axvline(0.8, color=axis, lw=1, ls=":")
    ax.set_xticks([0.55, 1, 2, 3, 4, 5]); ax.set_xticklabels(["best\nseed ◇", "1", "2", "3", "4", "5"])
    ax.set_xlim(0.3, 5.3); ax.set_xlabel("Number of models averaged", color=ink2); ax.set_ylabel(lab + "   ↓ better", color=ink2)
    ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle("Averaging models helps every setting; picking the best seed does not", x=0.01, ha="left", fontsize=14, fontweight="bold", color=ink)
fig.text(0.01, 0.885, "120 runs · 5 seeds on the same split × 3 splits × random/held-out · mean over splits · vs. the average single model "
         "(5 better in 24/24 splits) · ◇ = single model with the best val R_tot", color=ink2, fontsize=9.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.86), ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.8))
fig.savefig(os.path.join(HERE, "ensemble_results.png"), dpi=160, facecolor=bg)
