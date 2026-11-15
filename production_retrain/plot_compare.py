"""Old production vs retrained models (delivery val set): production_retrain/retrain_results.png"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
c = json.load(open(os.path.join(HERE, "comparison.json")))
KEYS = [("point_cgcnn", "CGCNN · point"), ("planar_cgcnn", "CGCNN · planar"),
        ("point_transformer", "Transformer · point"), ("planar_transformer", "Transformer · planar")]
ink, ink2, grid, axis, bg = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
old_c, new_c = "#8a8983", "#2a78d6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=bg, sharey=True)
for ax, (m, lab) in zip(axes, [("val_mae", "Validation per-atom MAE (meV/atom)"),
                               ("val_r_tot_graph_median", "Validation R_tot median, graph-sum (%)")]):
    ax.set_facecolor(bg)
    for i, (k, name) in enumerate(KEYS):
        y = len(KEYS) - 1 - i; d = c[k][m]
        ax.plot([d["old"], d["new_mean"]], [y, y], color=axis, lw=2, zorder=1)
        ax.plot(d["new"], [y] * len(d["new"]), "o", ms=4, color=new_c, alpha=0.4, zorder=2)
        ax.plot(d["old"], y, "o", ms=9, color=old_c, mec=bg, mew=1.5, zorder=3, label="old production" if i == 0 else None)
        ax.plot(d["new_mean"], y, "o", ms=9, color=new_c, mec=bg, mew=1.5, zorder=3, label="retrained (mean of 3 seeds)" if i == 0 else None)
        ax.annotate(f"{100 * (d['new_mean'] - d['old']) / d['old']:+.0f} %", (max(d["old"], d["new_mean"]), y),
                    xytext=(8, 0), textcoords="offset points", va="center", color=ink2, fontsize=9)
    ax.set_yticks(range(len(KEYS))); ax.set_yticklabels([n for _, n in KEYS[::-1]])
    ax.set_xlim(0, None); ax.margins(x=0.15)
    ax.grid(axis="x", color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel(lab + "   ← better", color=ink2)
fig.suptitle("Retrained with all new defaults: every production model improves", x=0.01, ha="left", fontsize=14, fontweight="bold", color=ink)
fig.text(0.01, 0.885, "Same delivery train/val split as production · small dots = individual seeds · val set also used for checkpoint selection (old and new)",
         color=ink2, fontsize=9.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.86), ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.83))
fig.savefig(os.path.join(HERE, "retrain_results.png"), dpi=160, facecolor=bg)
