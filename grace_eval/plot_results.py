"""GRACE C15 evaluation figure: grace_eval/grace_results.png"""
import json, os
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = torch.load(os.path.join(HERE, "graphs_relax.pt"), weights_only=False)
P = torch.load(os.path.join(HERE, "predictions.pt"), weights_only=False)
res = json.load(open(os.path.join(HERE, "results.json")))
ink, ink2, grid, axis, bg = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
col = {"cgcnn-ensemble5": "#2a78d6", "transformer-ensemble5": "#eb6834", "zero": "#8a8983"}
name = {"cgcnn-ensemble5": "CGCNN (5-model ensemble)", "transformer-ensemble5": "Transformer (5-model ensemble)", "zero": '"no relaxation" baseline'}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": axis, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2})
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), facecolor=bg, gridspec_kw={"width_ratios": [1.15, 1, 1]})
for ax in axes:
    ax.set_facecolor(bg); ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# A: FFHFF predicted vs true relaxation energy
ax = axes[0]
ii = [i for i, g in enumerate(R) if g.stack_sequence == "FFHFF"]
t = np.array([float(R[i].y.sum()) / R[i].num_nodes * 1000 for i in ii])
for m in ("cgcnn-ensemble5", "transformer-ensemble5"):
    p = np.array([float(P[m]["relax"][i].sum()) / R[i].num_nodes * 1000 for i in ii])
    ax.plot(t, p, "o", ms=5, color=col[m], alpha=0.7, mec=bg, mew=0.6, label=name[m])
lim = [-500, 20]; ax.plot(lim, lim, color=ink2, lw=1); ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("GRACE relaxation energy (meV/atom)"); ax.set_ylabel("predicted relaxation energy (meV/atom)")
ax.set_title("A  Stacking fault (FFHFF), 0 % strain: 189 compounds", loc="left", color=ink, fontsize=11)
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.text(-480, -15, "outliers: SnMg2, GeMg2, SiMg2, CoTi2, MgIn2\n(C15 far from stable in GRACE)", color=ink2, fontsize=8.5)

# B: median total-energy error per set vs zero baseline
ax = axes[1]
sets = [("FFHFF/all", "Stacking fault\n(FFHFF)"), ("Primitive/all", "Pristine C15,\nvolume relaxation"), ("XYZ/all", "Pristine C15,\nalready relaxed")]
w = 0.26
for k, m in enumerate(("zero", "cgcnn-ensemble5", "transformer-ensemble5")):
    vals = [res["relax"][s]["baseline_zero" if m == "zero" else m]["tot_err_median_meV_per_atom"] for s, _ in sets]
    xs = np.arange(len(sets)) + (k - 1) * w
    ax.bar(xs, vals, w * 0.92, color=col[m], label=name[m])
    for x, v in zip(xs, vals): ax.text(x, v + 0.08, f"{v:.2f}", ha="center", fontsize=8, color=ink2)
ax.set_xticks(range(len(sets))); ax.set_xticklabels([l for _, l in sets])
ax.set_ylabel("median error of total relaxation energy (meV/atom)")
ax.set_title("B  Relaxation at 0 % strain", loc="left", color=ink, fontsize=11); ax.legend(frameon=False, fontsize=9)

# C: strain test
ax = axes[2]
st = ["n10", "n5", "p0", "p5", "p10"]; xs = [-10, -5, 0, 5, 10]
for m, lab, c in (("input (no model)", "input energy (no model)", col["zero"]), ("transformer-ensemble5", name["transformer-ensemble5"], col["transformer-ensemble5"]),
                  ("cgcnn-ensemble5", name["cgcnn-ensemble5"], col["cgcnn-ensemble5"])):
    v = [res["strain"][f"FFHFF/{m}"]["abs_error_median_meV_per_atom"][s] for s in st]
    ax.plot(xs, v, "-o", color=c, lw=2, ms=6, mec=bg, label=lab)
ax.axhline(0, color=ink2, lw=1)
ax.set_xlabel("applied strain (%)"); ax.set_ylabel("|predicted - true relaxed energy| (meV/atom), median")
ax.set_title("C  Strain test (FFHFF), ideal = 0 at every strain", loc="left", color=ink, fontsize=11)
ax.text(0, 470, "all three curves overlap:\nthe models predict almost no\nrelaxation from strained cells", ha="center", color=ink2, fontsize=9)
ax.legend(frameon=False, fontsize=9)
fig.suptitle("GRACE C15 evaluation: stacking-fault relaxation transfers, strain and volume relaxation do not", x=0.01, ha="left", fontsize=14, fontweight="bold", color=ink)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(HERE, "grace_results.png"), dpi=150, facecolor=bg)
