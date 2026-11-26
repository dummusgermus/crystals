"""Relaxation-energy accuracy: deployed CGCNN / Graph Transformer (retrained on our splits, train_gnn_split.py)
vs the unified PaiNN (runs_unified, production config), on IDENTICAL test structures.

    python displacement/energy_compare/compare.py   -> energy_compare/results.md, plots/report/fig7_energy_vs_gnn.{png,pdf}

Per split (3 union compound splits) the common test set = structures present in every model's test output, stable
under our flags (point: data/superset_point_exact.pt unstable; planar: superset unstable). Labels: point = exact
full-cell LAMMPS ΔE (dE_exact), planar = full-cell ΔE (the cell is the graph). Metric: median |err| / |true|
(|true| > 1 meV) and MAE; single models = mean +- std over 3 splits x 2 seeds; ensemble = 2-seed mean per split,
mean over splits.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); DISP = os.path.dirname(HERE)
PRE = "joint_painn_rt8.9_buf4_k3_hk3_base+cyc+force_st1_eqvF_l1_norm_std_ema_nodf_ew0.3_uni_mx_eh_dfrecipe"
MODELS = ["CGCNN", "Graph Transformer", "unified PaiNN"]
EX = {(g.folder, g.defect_key): g for g in torch.load(os.path.join(DISP, "data", "superset_point_exact.pt"), weights_only=False)}


def painn(sp, seed):
    f = os.path.join(DISP, "runs_unified", "preds", f"{PRE}_s{seed}_sp{sp}.pt")
    if not os.path.isfile(f):
        return None
    out = {}
    for it in torch.load(f, weights_only=False, map_location="cpu")["test"]:
        if it["dom"] > 1:
            continue
        dom = ("point", "planar")[it["dom"]]
        out[(dom, it["folder"], it["key"])] = dict(pred=float(it["e_pred"].sum()), true_ours=float(it["e_true"].sum()),
                                                   unstable=bool(it["unstable"]))
    return out


def gnn(kind, sp, seed):
    out = {}
    for dom in ("point", "planar"):
        f = os.path.join(HERE, "preds", f"{dom}_{kind}_sp{sp}_s{seed}.json")
        if not os.path.isfile(f):
            return None
        for o in json.load(open(f))["test"]:
            out[(dom, o["folder"], o["key"])] = dict(pred=o["pred"], full_their=o["full_true"])
    return out


def label(k, pa):
    dom, folder, key = k
    if dom == "point":
        g = EX[(folder, key)]
        return float(g.dE_exact), bool(g.unstable)
    return pa[k]["true_ours"], pa[k]["unstable"]


def metrics(p, t):
    p, t = np.asarray(p), np.asarray(t); big = np.abs(t) > 1e-3
    return dict(rel=float(np.median(np.abs(p - t)[big] / np.abs(t[big]))), mae=float(np.mean(np.abs(p - t))), n=int(big.sum()))


def main():
    rows = {m: {d: [] for d in ("point", "planar")} for m in MODELS}
    ens = {m: {d: [] for d in ("point", "planar")} for m in MODELS}
    nset = {}
    for sp in range(3):
        runs = {s: {"CGCNN": gnn("cgcnn", sp, s), "Graph Transformer": gnn("transformer", sp, s),
                    "unified PaiNN": painn(sp, s)} for s in (0, 1)}
        if any(v is None for r in runs.values() for v in r.values()):
            print(f"split {sp}: incomplete, skipped"); continue
        pa = runs[0]["unified PaiNN"]
        common = set.intersection(*[set(v) for r in runs.values() for v in r.values()])
        for dom in ("point", "planar"):
            ks = sorted(k for k in common if k[0] == dom)
            lab = [label(k, pa) for k in ks]
            ks = [k for k, (t, u) in zip(ks, lab) if not u]; t = [label(k, pa)[0] for k in ks]
            nset[(sp, dom)] = len(ks)
            for m in MODELS:
                ps = [[runs[s][m][k]["pred"] for k in ks] for s in (0, 1)]
                rows[m][dom] += [metrics(p, t) for p in ps]
                ens[m][dom].append(metrics(np.mean(ps, 0), t))
        # label consistency check: their full-cell ΔE vs ours (planar identical, point dump-sum vs exact)
        for dom in ("point", "planar"):
            ks = [k for k in common if k[0] == dom]
            d = [abs(runs[0]["CGCNN"][k]["full_their"] - label(k, pa)[0]) for k in ks]
            print(f"split {sp} {dom}: common {len(ks)}, stable scored {nset[(sp, dom)]}, "
                  f"|their full ΔE - ours| median {np.median(d) * 1e3:.2f} meV, max {np.max(d) * 1e3:.1f} meV")

    L = ["# Relaxation energy: deployed CGCNN / Graph Transformer vs unified PaiNN", "",
         "Same test structures (our 3 union compound splits, test compounds unseen), same full-cell labels "
         "(point: exact LAMMPS ΔE, planar: cell ΔE), stable structures. CGCNN / GT: deployed recipe "
         "(delivery_global_v2, residual per-atom target, 2000 epochs) retrained on our train/val compounds. "
         "PaiNN: unified production config from runs_unified (λ 0.3 + harmonic residual, mixed-trained).", "",
         "Test structures per split: " + ", ".join(f"sp{sp} point {nset[(sp, 'point')]} / planar {nset[(sp, 'planar')]}"
                                                   for sp in range(3) if (sp, "point") in nset), "",
         "| model | point: median rel. error | point: MAE (meV) | planar: median rel. error | planar: MAE (meV) |",
         "|---|---|---|---|---|"]
    res = {}
    for title, src in (("single model (3 splits × 2 seeds)", rows), ("2-seed ensemble (mean over 3 splits)", ens)):
        L.append(f"| **{title}** | | | | |")
        for m in MODELS:
            c = []
            for dom in ("point", "planar"):
                r = [x["rel"] for x in src[m][dom]]; a = [x["mae"] * 1e3 for x in src[m][dom]]
                c += [f"{100 * np.mean(r):.1f}% ± {100 * np.std(r):.1f}", f"{np.mean(a):.0f} ± {np.std(a):.0f}"]
                res[(title, m, dom)] = (np.mean(r), np.std(r))
            L.append(f"| {m} | " + " | ".join(c) + " |")
    open(os.path.join(HERE, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    json.dump({f"{t}|{m}|{d}": v for (t, m, d), v in res.items()}, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    plot(res)


def plot(res):
    os.environ.setdefault("MPLCONFIGDIR", "/var/tmp/claude-2864/-work-log1-erik-mueller-code/33abe557-bd5a-401e-b4d0-c6af7d6bb3b3/scratchpad/mpl")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"CGCNN": "#E69F00", "Graph Transformer": "#56B4E9", "unified PaiNN": "#009E73"}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
    for ax, title in zip(axs, ("single model (3 splits × 2 seeds)", "2-seed ensemble (mean over 3 splits)")):
        w = 0.26
        for j, m in enumerate(MODELS):
            v = [100 * res[(title, m, d)][0] for d in ("point", "planar")]
            e = [100 * res[(title, m, d)][1] for d in ("point", "planar")]
            b = ax.bar(np.arange(2) + (j - 1) * w, v, w * 0.92, yerr=e, capsize=3, color=col[m], label=m)
            ax.bar_label(b, fmt="%.1f", padding=2, fontsize=8)
        ax.set_xticks(range(2)); ax.set_xticklabels(["point defects", "planar faults"])
        ax.set_title(title, fontsize=10); ax.set_ylabel("ΔE median relative error (%)")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Relaxation energy on identical unseen-compound test structures", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = os.path.join(DISP, "plots", "report", "fig7_energy_vs_gnn")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print("saved", out + ".png")


if __name__ == "__main__":
    main()
