"""Learning curve: GRACE-trained models (5-fold CV, 3-seed mean prediction) vs zero-shot. -> grace_train/learning_curve.png"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "analysis.json")))
NS = ["25", "50", "100", "all"]; X = [25, 50, 100, 128]
C = {"cgcnn": "#2a6fdb", "transformer": "#d9622b"}
panels = [("FFHFF", "tot_err_median", "Stacking fault (FFHFF): median error"),
          ("FFHFF", "tot_err_mean", "Stacking fault (FFHFF): mean error (outliers)"),
          ("Primitive", "tot_err_median", "Pristine C15, volume relaxation: median"),
          ("XYZ", "tot_err_median", "Already relaxed C15 (true = 0): median")]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, (t, k, title) in zip(axs, panels):
    for m in ("cgcnn", "transformer"):
        y = [res[f"{m}/n{n}/ckpt_rtot"]["seed_ensemble"][t][k] for n in NS]
        ax.plot(X, y, "o-", color=C[m], lw=2, ms=7, label=f"{m.upper() if m=='cgcnn' else 'Transformer'}, trained on GRACE")
        z = res[f"zero-shot/{m}-ensemble5"][t][k]
        ax.axhline(z, color=C[m], ls="--", lw=1.5, label=f"{'CGCNN' if m=='cgcnn' else 'Transformer'}, zero-shot (old data)")
    zb = res["zero-shot/transformer-fast"][t]["zero_baseline_median"] if k == "tot_err_median" else None
    if zb is not None and zb > 0:
        ax.axhline(zb, color="0.5", ls=":", lw=1.5, label="no relaxation")
    if k == "tot_err_mean":
        ax.set_yscale("log"); ax.set_yticks([0.8, 1, 2, 5, 10]); ax.set_yticklabels(["0.8", "1", "2", "5", "10"])
    ax.set_xticks(X); ax.set_xticklabels(["25", "50", "100", "128\n(all)"])
    ax.set_xlabel("GRACE training compounds"); ax.set_ylabel("error of total relaxation energy (meV/atom)")
    ax.set_title(title, fontsize=10); ax.set_ylim(bottom=0 if k != "tot_err_mean" else None); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8, loc="upper right")
fig.suptitle("GRACE-only training (5-fold CV by compound, 189 compounds, mean of 3 seeds) vs. zero-shot production models", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "learning_curve.png"), dpi=150)
