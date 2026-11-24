"""Plot runs_unified/unified_study.json -> plots/unified_mixed.png (python displacement/plot_unified.py)."""
import json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{HERE}/runs_unified/unified_study.json"))
cfg = [("PaiNN zero-shot ew0.3", "PaiNN, no mixed in training"), ("DefiNet zero-shot pos-only", "DefiNet, no mixed in training"),
       ("PaiNN mixed-trained ew0.3", "PaiNN, mixed in training"),
       ("PaiNN mixed-trained ew0.3 harm-res", "PaiNN, mixed in training, harmonic-residual energy"),
       ("DefiNet mixed-trained pos-only", "DefiNet, mixed in training")]
col = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#CC79A7"]
fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[5, 3]))
for a, keys, names, yl in (
        (ax[0], ["point_r_disp", "planar_r_disp", "mixed_r_disp_all", "mixed_pd_r_disp", "mixed_none_r_disp"],
         ["point", "planar", "mixed:\nall atoms", "mixed:\naround antisite", "mixed:\npure fault"], "r_disp (median |err|/|true|, test)"),
        (ax[1], ["point_e_rel", "planar_e_rel", "mixed_e_rel"], ["point", "planar", "mixed"], "relaxation-energy rel. error (test)")):
    x = np.arange(len(keys)); w = 0.8 / len(cfg)
    for i, (c, lab) in enumerate(cfg):
        v = [R[c][k] for k in keys]; e = [R[c][k + "_std"] for k in keys]
        if all(np.isnan(v)):
            continue
        a.bar(x + (i - (len(cfg) - 1) / 2) * w, v, w * 0.9, yerr=e, color=col[i], label=lab, capsize=2, error_kw=dict(lw=0.8))
    a.set_xticks(x); a.set_xticklabels(names, fontsize=9); a.set_ylabel(yl); a.grid(axis="y", alpha=0.3); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ax[0].axhline(0.426, color="0.4", ls="--", lw=1)
ax[0].text(-0.4, 0.437, "old domain-flag model on mixed cells, all atoms (0.426)", fontsize=8, color="0.3", ha="left")
ax[0].set_title("Displacements: one model for point, planar and mixed cells", fontsize=10)
ax[1].set_title("Energy (energy head; DefiNet runs had none)", fontsize=10)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.text(0.5, 0.005, "mean +/- std over 3 compound splits x 2 seeds, single models", ha="center", fontsize=8, color="0.4")
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(f"{HERE}/plots/unified_mixed.png", dpi=150)
