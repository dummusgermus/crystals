"""arch_sweep/arch_results.png: paired change vs. current architecture, per model."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
NICE = {"rbf16": "RBF 16 distances", "rbf32": "RBF 32 distances", "L2": "2 layers", "L3": "3 layers",
        "L4": "4 layers", "L6": "6 layers", "H64": "width 64", "H256": "width 256",
        "L4-H256": "4 layers, width 256", "L4-rbf16": "4 layers + RBF 16", "L6-rbf16": "6 layers + RBF 16"}
def main():
    rk = json.load(open(os.path.join(HERE, "ranking_variant.json")))
    ink, ink2, grid, axis, bg = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    blue, gray = "#2a78d6", "#8a8984"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": axis, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), facecolor=bg, sharex="col",
                             gridspec_kw={"height_ratios": [9, 7], "hspace": 0.25, "wspace": 0.06})
    for r, (model, title) in enumerate((("cgcnn", "CGCNN (current: 2 layers, width 128, raw distance)"),
                                        ("transformer", "Transformer (current: 4 layers, width 128, raw distance)"))):
        rows = sorted(rk[model], key=lambda x: -x["mean_d_rtot_pp"])
        for c, (key, pk, lab) in enumerate((("mean_d_rtot_pp", "p_rtot", "Δ test R_tot median (pp)"),
                                             ("mean_d_mae_pct", "p_mae", "Δ test MAE (%)"))):
            ax = axes[r, c]; ax.set_facecolor(bg)
            for y, row in enumerate(rows):
                v = row[key]; good = v < 0 and row[pk] < 0.05
                ax.barh(y, v, height=0.6, color=blue if good else gray, zorder=2)
                ax.text(v + (0.4 if v >= 0 else -0.4), y, f"{v:+.1f}", va="center",
                        ha="left" if v >= 0 else "right", color=ink2, fontsize=9)
            ax.axvline(0, color=ink2, lw=1.2, zorder=3); ax.grid(axis="x", color=grid, lw=0.8); ax.set_axisbelow(True)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([NICE[x["variant"]] for x in rows] if c == 0 else [""] * len(rows))
            if r == 1: ax.set_xlabel(lab + "   ← better", color=ink2)
            if c == 0: ax.set_title(title, loc="left", color=ink, fontsize=11, fontweight="bold")
    for ax in axes[:, 0]: lo, hi = ax.get_xlim(); ax.set_xlim(lo - 3, hi + 3)
    for ax in axes[:, 1]: lo, hi = ax.get_xlim(); ax.set_xlim(lo - 5, hi + 5)
    fig.suptitle("Architecture sweep: CGCNN gains from 3–4 layers; the Transformer is already at its best size",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=ink)
    fig.text(0.01, 0.935, "216 runs · new training recipe · paired with the current architecture (same split + seed), "
             "3 seeds × point/planar × random/held-out · blue = significantly better (p < 0.05)", color=ink2, fontsize=10)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.87, bottom=0.08)
    fig.savefig(os.path.join(HERE, "arch_results.png"), dpi=160, facecolor=bg)
if __name__ == "__main__":
    main()
