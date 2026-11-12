"""plateau_study/plateau_results.png: (a) paired change vs. current recipe per variant,
(b) val-MAE curves of the current recipe vs. combo-rob (random split, all seeds)."""
import glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NICE = {"rob": "robust target scaling", "lr1e-3": "LR 1e-3", "lr5e-4": "LR 5e-4", "clip": "gradient clipping",
        "pat30": "patient LR scheduler", "cos": "cosine schedule (LR 2e-3)", "msewu": "MSE warm-up",
        "bn": "batch norm", "combo": "LR 1e-3 + clip + cosine", "combo-rob": "combo + robust scaling",
        "base": "safer checkpoint only"}
def main():
    an = json.load(open(os.path.join(HERE, "analysis.json")))
    order = sorted(an, key=lambda v: -an[v]["d_mae_pct"])
    ink, ink2, grid, axis, bg = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    blue, orange, gray = "#2a78d6", "#eb6834", "#8a8984"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": axis, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink})
    fig = plt.figure(figsize=(15, 6.6), facecolor=bg)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.08)
    a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1], sharey=a0); a2 = fig.add_subplot(gs[2])
    for ax, key, lab in ((a0, "d_rtot_pp", "Δ test R_tot median (pp)"), (a1, "d_mae_pct", "Δ test MAE (%)")):
        ax.set_facecolor(bg)
        for y, v in enumerate(order):
            good = an[v][key] < 0 and an[v]["p_rtot" if key == "d_rtot_pp" else "p_mae"] < 0.05
            ax.barh(y, an[v][key], height=0.6, color=blue if good else gray, zorder=2)
            ax.text(an[v][key] + (1 if an[v][key] >= 0 else -1), y, f"{an[v][key]:+.0f}", va="center",
                    ha="left" if an[v][key] >= 0 else "right", color=ink2, fontsize=9)
        ax.axvline(0, color=ink2, lw=1.2, zorder=3); ax.grid(axis="x", color=grid, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.set_xlabel(lab + "   ← better", color=ink2)
        lo, hi = ax.get_xlim(); ax.set_xlim(lo - 8, hi + 8)
    a0.set_yticks(range(len(order))); a0.set_yticklabels([NICE[v] for v in order])
    plt.setp(a1.get_yticklabels(), visible=False)
    a2.set_facecolor(bg)
    for v, c, lab in (("base", orange, "current recipe"), ("combo-rob", blue, "combo + robust scaling")):
        for i, f in enumerate(sorted(glob.glob(os.path.join(HERE, "results", f"random_planar_cgcnn_*_v-{v}_s*.json")))):
            vm = 1000 * np.array(json.load(open(f))["curves"]["val_mae"])
            a2.plot(np.arange(1, len(vm) + 1), vm, color=c, lw=1.2, alpha=0.75, label=lab if i == 0 else None)
    a2.set_yscale("log"); a2.set_xscale("log"); a2.set_ylim(3, 80)
    a2.set_yticks([4, 6, 10, 20, 40]); a2.set_yticklabels(["4", "6", "10", "20", "40"])
    a2.axhspan(28, 40, color=grid, alpha=0.6, zorder=0); a2.text(1.2, 41, "plateau (≈ predict the median)", color=ink2, fontsize=9)
    a2.grid(color=grid, lw=0.8); a2.set_axisbelow(True)
    for s in ("top", "right"): a2.spines[s].set_visible(False)
    a2.set_xlabel("epoch", color=ink2); a2.set_ylabel("validation MAE (meV/atom)", color=ink2)
    a2.legend(frameon=False, loc="lower left"); a2.set_title("Planar CGCNN, random split, 6 seeds each", color=ink2, fontsize=10, loc="left")
    fig.suptitle("Plateau fix: lower LR + gradient clipping + cosine schedule (+ robust scaling) halves the planar CGCNN error",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", color=ink)
    fig.text(0.01, 0.915, "132 runs · paired with the current recipe (same split + seed), 6 seeds × random/held-out chemistry · "
             "blue = significantly better (p < 0.05) · variants evaluated at the safer checkpoint", color=ink2, fontsize=10)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.86, bottom=0.1)
    fig.savefig(os.path.join(HERE, "plateau_results.png"), dpi=160, facecolor=bg)
if __name__ == "__main__":
    main()
