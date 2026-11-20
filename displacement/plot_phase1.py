"""Phase-1 overview: val/test r_disp per config (mean over 2 seeds), CGCNN vs Transformer."""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
runs = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs")
A = json.load(open(os.path.join(runs, "analysis_phase1.json")))
C = {"cgcnn": "#2a6fdb", "transformer": "#e07b39", "painn": "#3a9e5f"}

def label(r):
    v = r["view"]; s = []
    if r["domain"] == "point" and (v["rt"], v["buf"]) != (8.9, 4): s.append(f"rt{v['rt']:g}/buf{v['buf']:g}")
    if v["edge"] != "k3": s.append(v["edge"])
    f = v["feat"].replace("base+cyc", "").replace("base", "-cyc").strip("+")
    if f: s.append(f)
    if r["head"] != "eqv" or r["head_force"]: s.append("head " + r["head"] + ("+F·u" if r["head_force"] else ""))
    if r["loss"] != "mse": s.append(r["loss"])
    if v["stable"]: s.append("stable-only")
    return ", ".join(s) or "centre (k3, base+cyc)"

fig, axs = plt.subplots(1, 2, figsize=(15, 9))
for ax, dom in zip(axs, ("point", "planar")):
    rows = [r for r in A["rows"] if r["domain"] == dom and r["model"] in ("cgcnn", "transformer")]
    labs = sorted({label(r) for r in rows},
                  key=lambda l: min(r["val_r_disp"] for r in rows if label(r) == l and r["model"] == "cgcnn") if any(label(r) == l and r["model"] == "cgcnn" for r in rows) else 9)
    y = {l: i for i, l in enumerate(labs)}
    for r in rows:
        off = -0.15 if r["model"] == "cgcnn" else 0.15
        yy = y[label(r)] + off
        ax.plot([r["val_r_disp"]], [yy], "o", color=C[r["model"]], ms=6)
        ax.plot([r["test_r_disp"]], [yy], "s", mfc="none", color=C[r["model"]], ms=6)
    b = A["force_baseline"][dom]["test"]["r_disp"]
    ax.axvline(b, color="0.4", ls="--", lw=1); ax.text(b, len(labs) - 0.3, " d = c·F0", color="0.3", fontsize=8)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 1.0); ax.grid(axis="x", alpha=.3)
    ax.set_xlabel("r_disp = |error| / |true displacement| (median per structure; 1 = no motion)")
    ax.set_title(f"{dom} defects")
from matplotlib.lines import Line2D
axs[1].legend(handles=[Line2D([], [], marker="o", ls="", color=C["cgcnn"], label="CGCNN val"),
                       Line2D([], [], marker="s", ls="", mfc="none", color=C["cgcnn"], label="CGCNN test"),
                       Line2D([], [], marker="o", ls="", color=C["transformer"], label="Transformer val"),
                       Line2D([], [], marker="s", ls="", mfc="none", color=C["transformer"], label="Transformer test")],
              loc="lower right", fontsize=8)
fig.suptitle("Displacement prediction, phase 1 screen (mean of 2 seeds, one compound split; sorted by CGCNN val)")
fig.tight_layout(); out = os.path.join(runs, "phase1_overview.png"); fig.savefig(out, dpi=130); print(out)
