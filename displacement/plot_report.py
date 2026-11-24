"""Summary figures for the report -> displacement/plots/report/*.png (+ .pdf).

    MPLCONFIGDIR=... python displacement/plot_report.py

fig1_unified_vs_separate  one multi-task model vs separate position / energy models (positions, energy, runtime)
fig2_vs_definet_e3relax   our models vs the authors' DefiNet / E3Relax / E3Relax-H2: on our data and on their benchmarks
fig3_positions            production ensemble on held-out compounds: parity, error vs distance, per-structure error, direction
fig4_examples             predicted vs true displacement field: one point defect, one planar fault
fig5_energy               production ensemble relaxation energy on held-out compounds
fig6_data_generation      our GRACE pipeline vs the team's LAMMPS labels (pilot + rattle test)
Sources: runs_unified, runs_separate, runs_ext, runs_joint, runs_final_unified (saved test items), bench_data,
gen_data; numbers quoted from RESULTS.md where no per-run file exists (external benchmark table, runtime).
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "plots", "report")
sys.path.insert(0, HERE)
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          verm="#D55E00", purple="#CC79A7", grey="#999999")
DOM = {"point": OI["blue"], "planar": OI["orange"], "mixed": OI["green"]}
DOMS = ("point", "planar", "mixed")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig); print("wrote", name)


def legend_below(fig, handles=None, labels=None, ncol=4, y=-0.02):
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol)


def runs(pattern, keys):
    out = []
    for f in glob.glob(os.path.join(HERE, pattern)):
        t = json.load(open(f))["test"]; out.append([t.get(k, np.nan) for k in keys])
    return np.array(out, dtype=float)


# ----------------------------------------------------------------------------------------------- fig 1
def fig1():
    R = ["point_r_disp", "planar_r_disp", "mixed_r_disp_all"]; E = ["point_e_rel", "planar_e_rel", "mixed_e_rel"]
    P = "joint_painn_rt8.9_buf4_k3_hk3_base+cyc+force_st1_eqvF_l1_norm_std_ema_nodf_"
    cfg = {  # label: (results glob, colour)
        "separate: position-only": (f"runs_separate/results/{P}uni_mx_dfrecipe_s*_sp*.json", OI["sky"]),
        "separate: energy-only": (f"runs_separate/results/{P}ew1_pw0_uni_mx_dfrecipe_s*_sp*.json", OI["grey"]),
        "separate: energy-only + harmonic": (f"runs_separate/results/{P}ew1_pw0_uni_mx_eh_dfrecipe_s*_sp*.json", OI["purple"]),
        "unified: multi-task": (f"runs_unified/results/{P}ew0.3_uni_mx_dfrecipe_s*_sp*.json", OI["orange"]),
        "unified: multi-task + harmonic residual": (f"runs_unified/results/{P}ew0.3_uni_mx_eh_dfrecipe_s*_sp*.json", OI["verm"]),
    }
    data = {}
    for k, (pat, _) in cfg.items():
        a = runs(pat, R + E)
        if not len(a):
            fs = glob.glob(os.path.join(HERE, os.path.dirname(pat), "*.json"))
            raise SystemExit(f"no runs for {k}: {pat} (have e.g. {[os.path.basename(f) for f in fs[:4]]})")
        data[k] = a
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 0.8]), constrained_layout=True)
    for ax, cols, title, ylab in ((axs[0], slice(0, 3), "Positions", "r_disp (test, lower = better)"),
                                  (axs[1], slice(3, 6), "Relaxation energy", "median relative error (test)")):
        ks = [k for k in cfg if np.isfinite(data[k][:, cols]).all() and data[k][:, cols].max() < 0.9]
        w = 0.8 / len(ks)
        for j, k in enumerate(ks):
            m, s = data[k][:, cols].mean(0), data[k][:, cols].std(0)
            x = np.arange(3) + (j - (len(ks) - 1) / 2) * w
            ax.bar(x, m, w * 0.92, yerr=s, color=cfg[k][1], label=k, error_kw=dict(lw=0.8, capsize=2),
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(DOMS); ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
    # runtime (bench_runtime.py, L4 GPU, mixed cell 292 atoms, batch 1)
    ax = axs[2]
    feat, prep, fwd = 2386, 3.9, 14.9
    rows = [("one unified model", [feat, prep, fwd]), ("two separate models", [feat, prep, 2 * fwd])]
    for i, (lab, parts) in enumerate(rows):
        left = 0
        for v, c, n in zip(parts, (OI["grey"], OI["sky"], OI["verm"]), ("graph construction (CPU)", "input preparation", "model forward (GPU)")):
            ax.barh(i, v, left=left, color=c, height=0.55, label=n if i == 0 else None, edgecolor="white", linewidth=0.5)
            left += v
        ax.text(left + 30, i, f"{left / 1e3:.2f} s", va="center")
        ax.text(0, i - 0.36, lab, va="bottom", fontsize=8.5)
    ax.set_yticks([]); ax.set_ylim(1.6, -0.75)
    ax.set_xlabel("time per cell (ms)"); ax.set_xlim(0, 2900); ax.grid(axis="y", visible=False)
    ax.set_title("Inference time, end to end", loc="left", fontweight="bold")
    ax.text(0.0, -0.2, f"model forward alone: {fwd:.0f} ms vs {2 * fwd:.0f} ms (+{100 * fwd / (prep + fwd):.0f}%)\n"
            f"end to end: +{100 * fwd / (feat + prep + fwd):.1f}%", transform=ax.transAxes, va="top", fontsize=8)
    h1, l1 = axs[0].get_legend_handles_labels(); h2, l2 = axs[1].get_legend_handles_labels()
    hs, ls = list(h1), list(l1)
    for h, l in zip(h2, l2):
        if l not in ls:
            hs.append(h); ls.append(l)
    fig.legend(hs, ls, loc="upper left", bbox_to_anchor=(0.04, -0.01), ncol=2)
    axs[2].legend(loc="upper left", bbox_to_anchor=(0.0, -0.36), ncol=1)
    fig.suptitle("One unified model vs separate position and energy models (mean ± std, 3 compound splits × 2 seeds)",
                 x=0.01, ha="left")
    save(fig, "fig1_unified_vs_separate")


# ----------------------------------------------------------------------------------------------- fig 2
def fig2():
    def r(pat):
        a = runs(pat, ["r_disp"]); return a.mean(), a.std()
    X = "rt8.9_buf4_r6n50_hr6n50_base+cyc+force_st1_eqv_l1"
    models = [  # label, point glob, planar glob, colour
        ("DefiNet (as published)", f"runs_ext/results/point_definet_{X}_h512_ema_s*", f"runs_ext/results/planar_definet_{X}_h512_ema_s*", OI["sky"]),
        ("E3Relax (as published)", f"runs_ext/results/point_e3relax_{X}_lw_h512_ema_s*", f"runs_ext/results/planar_e3relax_{X}_lw_h512_ema_s*", OI["blue"]),
        ("E3Relax-H2 (as published)", f"runs_ext/results/point_e3relaxh2_{X}_lw_h512_ema_s*", f"runs_ext/results/planar_e3relaxh2_{X}_lw_h512_ema_s*", OI["purple"]),
        ("DefiNet + our inputs", f"runs_ext/results/point_definet_{X}_std_xf_xF_h512_ema_s*", f"runs_ext/results/planar_definet_{X}_std_xf_xF_h512_ema_s*", OI["grey"]),
    ]
    vals = [(lab, r(p), r(q), c) for lab, p, q, c in models]
    P = "joint_painn_rt8.9_buf4_k3_hk3_base+cyc+force_st1_eqvF_l1_norm_std_ema_nodf_"
    u = runs(f"runs_unified/results/{P}ew0.3_uni_mx_eh_dfrecipe_s*_sp*.json", ["point_r_disp", "planar_r_disp"])
    vals.append(("ours: unified PaiNN (single model)", (u[:, 0].mean(), u[:, 0].std()), (u[:, 1].mean(), u[:, 1].std()), OI["orange"]))
    vals.append(("ours: ensemble PaiNN ×2 + DefiNet ×2", (0.129, np.nan), (0.125, np.nan), OI["verm"]))  # runs_joint ensemble, RESULTS.md
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[0.85, 1.15]))
    ax = axs[0]; w = 0.8 / len(vals)
    for j, (lab, p, q, c) in enumerate(vals):
        x = np.arange(2) + (j - (len(vals) - 1) / 2) * w
        m = [p[0], q[0]]; s = [p[1], q[1]]
        ax.bar(x, m, w * 0.92, yerr=None if np.isnan(s).all() else s, color=c, label=lab,
               error_kw=dict(lw=0.8, capsize=2), edgecolor="white", linewidth=0.5)
        for xi, mi, si in zip(x, m, s):
            ax.text(xi, mi + np.nan_to_num(si) + 0.006, f"{mi:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["point defects", "planar faults"])
    ax.set_ylabel("r_disp on unseen compounds (lower = better)"); ax.grid(axis="x", visible=False)
    ax.set_title("a  On our data", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    # benchmarks: test coordinate MAE / unrelaxed MAE (RESULTS.md, runs_bench)
    B = {  # dataset: unrelaxed, GRACE relax, authors' model, ours (PaiNN + GRACE + recipe), E3Relax-H2 paper
        "DefiNet low": (0.0243, 0.0071, 0.0017, 0.0017, np.nan), "DefiNet high": (0.0781, 0.0507, 0.0390, 0.0420, np.nan),
        "MP": (0.0965, 0.0844, 0.0565, 0.0518, 0.054), "JARVIS": (0.0550, 0.0693, 0.0407, 0.0405, 0.040),
        "vdW": (0.1508, 0.1814, 0.1164, 0.1125, 0.114), "XMnO": (0.3185, 0.2240, 0.1039, 0.1350, 0.092),
        "C2DB": (0.2688, 0.3293, 0.1683, 0.2306, 0.161)}
    ax = axs[1]; names = list(B)[::-1]
    for i, n in enumerate(names):
        u0, g, a, o, h = B[n]
        ax.plot([min(a, o) / u0, max(a, o) / u0], [i, i], color="#cccccc", lw=2, zorder=1)
        ax.scatter(g / u0, i, marker="x", color=OI["grey"], s=28, zorder=2, label="GRACE relaxation alone" if i == 0 else None)
        ax.scatter(a / u0, i, color=OI["blue"], s=34, zorder=3, label="authors' model (DefiNet / E3Relax, our run)" if i == 0 else None)
        if np.isfinite(h):
            ax.scatter(h / u0, i, facecolor="white", edgecolor=OI["purple"], s=34, lw=1.3, zorder=3,
                       label="E3Relax-H2 (paper)" if n == "C2DB" else None)
        ax.scatter(o / u0, i, color=OI["verm"], s=34, zorder=4, label="ours: PaiNN + GRACE inputs" if i == 0 else None)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.set_xlim(0, 1.3)
    ax.axvline(1, color=OI["black"], lw=0.8, ls="--"); ax.text(1.01, len(names) - 0.6, "unrelaxed\ninput", fontsize=7.5, va="top")
    ax.set_xlabel("test coordinate MAE / MAE of the unrelaxed input (lower = better)")
    ax.set_title("b  On the authors' benchmarks (their data and splits)", loc="left", fontweight="bold"); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    fig.text(0.01, -0.2, "a: test = compounds never seen in training, mean ± std over 3 compound splits × 2 seeds (authors' models "
             "on per-domain splits, ours on the joint point+planar split). b: mean of 3 seeds; GRACE-OMAT's training data "
             "overlaps with MP; JARVIS/vdW references use other functionals.", fontsize=7.5, wrap=True)
    save(fig, "fig2_vs_definet_e3relax")


# ------------------------------------------------------------------------------------- held-out items
FIN = "runs_final_unified/preds/"
MEMBERS = [f"{FIN}joint_painn_rt8.9_buf4_k3_hk3_base+cyc+force_st1_eqvF_l1_norm_std_ema_nodf_ew0.3_uni_mx_eh_final_dfrecipe_s{s}_sp0.pt" for s in (0, 1)] + \
          [f"{FIN}joint_definet_rt8.9_buf4_k3_hk3_base+cyc+force_st1_eqv_l1_std_xf_xF_h512_ema_nodf_uni_mx_final_s{s}_sp0.pt" for s in (0, 1)]


def heldout():
    """Production ensemble (= the 4 members in production/) on the held-out 10% of compounds, from the saved test items."""
    cache = os.path.join(OUT, "heldout_items.pt")
    if os.path.isfile(cache):
        return torch.load(cache, weights_only=False)
    M = [torch.load(os.path.join(HERE, f), weights_only=False, map_location="cpu")["test"] for f in MEMBERS]
    ids = [[(t["dom"], t["folder"], t["key"]) for t in m] for m in M]
    assert all(i == ids[0] for i in ids), "member test items not aligned"
    ex = {(g.folder, g.defect_key): g.dE_exact for g in torch.load(os.path.join(HERE, "data", "superset_point_exact.pt"), weights_only=False)}
    items = []
    for k, t in enumerate(M[0]):
        it = dict(dom=DOMS[t["dom"]], folder=t["folder"], key=t["key"], y=t["y"], mask=t["mask"], dist=t["dist"],
                  unstable=t["unstable"], pred=torch.stack([m[k]["pred"] for m in M]).mean(0),
                  e_pred=float(np.mean([float(m[k]["e_pred"].sum()) for m in M if "e_pred" in m[k]])),
                  e_true=float(t["e_true"].sum()))
        it["e_exact"] = ex.get((t["folder"], t["key"]), np.nan) if it["dom"] == "point" else it["e_true"]
        items.append(it)
    torch.save(items, cache)
    return items


def scored(it):
    m = it["mask"].clone()
    if it["dom"] == "point":
        m &= it["dist"] <= 8.9
    return it["y"][m].numpy(), it["pred"][m].numpy(), it["dist"][m].numpy()


def r_disp(it):
    y, p, _ = scored(it); n = np.linalg.norm(y)
    return np.linalg.norm(p - y) / n if n > 1e-4 else np.nan


# ----------------------------------------------------------------------------------------------- fig 3
def fig3():
    items = [it for it in heldout() if not it["unstable"]]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7.2))
    for ax, d in zip(axs[0], DOMS):
        Y, P = [], []
        for it in items:
            if it["dom"] == d:
                y, p, _ = scored(it); Y.append(y); P.append(p)
        Y = np.concatenate(Y); P = np.concatenate(P)
        ny, npd = np.linalg.norm(Y, axis=1), np.linalg.norm(P, axis=1); s = ny > 1e-3
        hb = ax.hexbin(ny[s], npd[s].clip(1e-4), xscale="log", yscale="log", gridsize=45, bins="log", cmap="Blues",
                       mincnt=1, extent=(-3, 0.5, -3, 0.5), linewidths=0)
        ax.plot([1e-3, 3], [1e-3, 3], color=OI["black"], lw=0.8, ls="--")
        ax.set_xlim(1e-3, 3); ax.set_ylim(1e-3, 3); ax.set_aspect("equal")
        ax.set_xlabel("true |displacement| (Å)"); ax.set_ylabel("predicted |displacement| (Å)")
        cos = (Y[s] * P[s]).sum(1) / (ny[s] * np.maximum(npd[s], 1e-12))
        big = ny[s] > 0.01
        ax.set_title(f"{'abc'[DOMS.index(d)]}  {d}: per-atom magnitude", loc="left", fontweight="bold")
        ax.text(0.04, 0.96, f"{s.sum():,} atoms\nmedian direction cosine\n(|d| > 0.01 Å): {np.median(cos[big]):.3f}",
                transform=ax.transAxes, va="top", fontsize=7.5)
        fig.colorbar(hb, ax=ax, shrink=0.75, label="atoms per bin")
    # d: point error vs distance
    ax = axs[1, 0]
    D, E, M = [], [], []
    for it in items:
        if it["dom"] == "point":
            y, p, dist = scored(it); D.append(dist); E.append(np.linalg.norm(p - y, axis=1)); M.append(np.linalg.norm(y, axis=1))
    D, E, M = map(np.concatenate, (D, E, M)); edges = np.arange(1.5, 9.6, 1.0); c = (edges[1:] + edges[:-1]) / 2  # defect atom (d=0) excluded
    q = lambda v, lo, hi, f: np.array([f(v[(D >= a) & (D < b)]) if ((D >= a) & (D < b)).any() else np.nan for a, b in zip(lo, hi)])  # noqa: E731
    for v, col, lab in ((M, OI["black"], "true |d|"), (E, OI["verm"], "|error|")):
        md = q(v, edges[:-1], edges[1:], np.median)
        lo_ = q(v, edges[:-1], edges[1:], lambda x: np.percentile(x, 25)); hi_ = q(v, edges[:-1], edges[1:], lambda x: np.percentile(x, 75))
        ax.plot(c, md, "-o", color=col, lw=2, ms=4, label=f"{lab} (median, IQR)")
        ax.fill_between(c, lo_, hi_, color=col, alpha=0.15, lw=0)
    ax.set_yscale("log"); ax.set_xlabel("distance from the point defect (Å)"); ax.set_ylabel("Å")
    ax.set_title("d  point defects: error vs distance", loc="left", fontweight="bold"); ax.legend(fontsize=8)
    # e: per-structure r_disp CDF
    ax = axs[1, 1]
    for d in DOMS:
        r = np.sort([r_disp(it) for it in items if it["dom"] == d]); r = r[np.isfinite(r)]
        ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=DOM[d], lw=2, label=f"{d}: median {np.median(r):.3f} (n={len(r)})")
    ax.axvline(1, color=OI["black"], lw=0.8, ls="--"); ax.text(0.97, 0.45, "no motion →", ha="right", fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlim(0.02, 1.5); ax.set_xlabel("per-structure r_disp = |error| / |true|")
    ax.set_ylabel("fraction of structures"); ax.set_title("e  per-structure error", loc="left", fontweight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=7.5)
    # f: fraction of displacement norm captured (skill) per domain, as bars with pooled numbers
    ax = axs[1, 2]
    for j, d in enumerate(DOMS):
        Y = np.concatenate([scored(it)[0] for it in items if it["dom"] == d]); P = np.concatenate([scored(it)[1] for it in items if it["dom"] == d])
        skill = 1 - ((P - Y) ** 2).sum() / (Y ** 2).sum()
        ax.bar(j, skill, color=DOM[d], width=0.6); ax.text(j, skill + 0.02, f"{skill:.2f}", ha="center")
    ax.set_xticks(range(3)); ax.set_xticklabels(DOMS); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    ax.set_ylabel("skill = 1 − Σ|err|² / Σ|d|²")
    ax.set_title("f  pooled skill (1 = perfect, 0 = no motion)", loc="left", fontweight="bold")
    n = {d: sum(it["dom"] == d for it in items) for d in DOMS}
    fig.suptitle(f"Production ensemble (PaiNN ×2 + DefiNet ×2) on held-out compounds: {n['point']} point, {n['planar']} planar, "
                 f"{n['mixed']} mixed cells (point: atoms within 8.9 Å)", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, "fig3_positions")


# ----------------------------------------------------------------------------------------------- fig 4
def predict(graphs, tag):
    p_in, p_out = os.path.join(OUT, f"_ex_{tag}_in.pt"), os.path.join(OUT, f"_ex_{tag}_out.pt")
    if not os.path.isfile(p_out):
        torch.save(graphs, p_in)
        subprocess.run([sys.executable, os.path.join(HERE, "predict_disp.py"), p_in, p_out], check=True,
                       stdout=subprocess.DEVNULL)
    return torch.load(p_out, weights_only=False)


def fig4():
    items = [it for it in heldout() if not it["unstable"]]
    pt = sorted([it for it in items if it["dom"] == "point" and it["key"].endswith("-1-2-8a") and np.isfinite(r_disp(it))], key=r_disp)
    pl = sorted([it for it in items if it["dom"] == "planar" and np.isfinite(r_disp(it))], key=r_disp)
    mid = pt[int(0.35 * len(pt)): int(0.65 * len(pt)) + 1]  # middle third of the error distribution
    ip = max(mid, key=lambda it: float(it["y"].norm(dim=-1).max()))  # ... with the clearest displacement field
    iq = pl[len(pl) // 2]
    G = {g.defect_key: g for g in torch.load(os.path.join(HERE, "data", "superset_point.pt"), weights_only=False) if g.folder == ip["folder"]}
    gp = G[ip["key"]]
    gq = next(g for g in torch.load(os.path.join(HERE, "data", "superset_planar.pt"), weights_only=False) if g.folder == iq["folder"])
    op = predict([gp], "point")[0]; oq = predict([gq], "planar")[0]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    # point: slab through the defect, perpendicular to [001]
    ids = op["node_ids"]; pos = gp.rel[ids].numpy(); dt = gp.disp[ids].numpy(); dp = op["disp"].numpy()
    sl = (np.abs(pos[:, 2]) < 1.0) & (np.linalg.norm(pos[:, :2], axis=1) < 8.5)
    sc = max(1, round(1.4 / max(np.linalg.norm(dt[sl, :2], axis=1).max(), 1e-3)))
    for ax, d, col, title in ((axs[0], dt, OI["black"], "true"), (axs[1], dp, OI["verm"], "predicted")):
        ax.scatter(pos[sl, 0], pos[sl, 1], s=14, c=[OI["blue"] if t == 1 else OI["orange"] for t in gp.type[ids][sl].tolist()], zorder=1)
        ax.quiver(pos[sl, 0], pos[sl, 1], d[sl, 0], d[sl, 1], color=col, angles="xy", scale_units="xy", scale=1 / sc,
                  width=0.006, zorder=2)
        ax.scatter([0], [0], marker="*", s=180, color=OI["yellow"], edgecolor=OI["black"], zorder=3)
        ax.set_aspect("equal"); ax.set_xlim(-9, 9); ax.set_ylim(-9, 9); ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
        ax.set_title(f"{'ab'[ax is axs[1]]}  point defect, {title} (arrows ×{sc:.0f})", loc="left", fontweight="bold")
    axs[0].text(0.01, -0.16, f"{ip['folder'][4:]} 8a antisite (held-out compound), ±1 Å slab through the defect (star); "
                f"r_disp {r_disp(ip):.2f} (middle third of held-out point cells). Blue: A atoms, orange: B atoms.",
                transform=axs[0].transAxes, fontsize=7.5, va="top")
    # planar: displacement along c vs height
    ax = axs[2]; ids = oq["node_ids"]; H = gq.cell0.view(3, 3).numpy()
    pos = gq.rel[ids].numpy() if hasattr(gq, "rel") else None
    yt = iq["y"].numpy(); yp = iq["pred"].numpy()
    cdir = H[2] / np.linalg.norm(H[2])
    z = (pos @ cdir) if pos is not None and len(pos) == len(yt) else np.arange(len(yt))
    o = np.argsort(z)
    ax.scatter(yt[o] @ cdir, z[o], s=10, color=OI["black"], label="true", zorder=2)
    ax.scatter(yp[o] @ cdir, z[o], s=10, facecolor="none", edgecolor=OI["verm"], lw=1, label="predicted", zorder=3)
    zf = z[(gq.is_defect[ids] > 0.5).numpy()]  # planar superset: is_defect = fault-layer atoms
    for zz in np.unique(np.round(zf, 1)):
        ax.axhline(zz, color=OI["orange"], lw=0.6, alpha=0.5, zorder=1)
    ax.set_xlabel("displacement along the stacking direction c (Å)"); ax.set_ylabel("height along c (Å)")
    ax.set_title("c  planar fault: displacement profile", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.text(0.01, -0.36, f"{iq['folder']} (held-out), r_disp {r_disp(iq):.2f} (median of held-out planar cells); "
            f"orange lines: fault-layer atoms.", transform=ax.transAxes, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig4_examples")


# ----------------------------------------------------------------------------------------------- fig 5
def fig5():
    items = [it for it in heldout() if not it["unstable"]]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, d in zip(axs, DOMS):
        t = np.array([it["e_exact"] for it in items if it["dom"] == d]); p = np.array([it["e_pred"] for it in items if it["dom"] == d])
        ok = np.isfinite(t) & (np.abs(t) > 1e-3)
        t, p = -t[ok], -p[ok]
        ax.scatter(t, np.clip(p, 1e-4, None), s=14, color=DOM[d], alpha=0.8, edgecolor="white", lw=0.3)
        lo, hi = 1e-3, max(t.max(), p.max()) * 1.5
        ax.plot([lo, hi], [lo, hi], color=OI["black"], lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        rel = np.abs(p - t) / np.abs(t)
        ax.text(0.04, 0.96, f"n = {len(t)}\nmedian rel. error {np.median(rel):.3f}\nMAE {1e3 * np.mean(np.abs(p - t)):.0f} meV",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("true relaxation energy −ΔE (eV)"); ax.set_ylabel("predicted −ΔE (eV)")
        ax.set_title(f"{'abc'[DOMS.index(d)]}  {d}", loc="left", fontweight="bold")
    fig.suptitle("Relaxation energy, production ensemble (PaiNN energy heads, harmonic residual) on held-out compounds",
                 x=0.01, ha="left", y=1.02)
    fig.text(0.01, -0.06, "Point defects against the exact full-cell LAMMPS totals (includes far-field relaxation outside "
             "the model's ~13 Å view); planar and mixed against the per-atom label sums.", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig5_energy")


# ----------------------------------------------------------------------------------------------- fig 6
def fig6():
    from make_point_exact import yaml_pe
    sim = os.path.join(os.path.dirname(HERE), "SIMULATIONS")
    D = [d for p in sorted(glob.glob(os.path.join(HERE, "gen_data", "pilot_part*.pkl"))) for d in pickle.load(open(p, "rb"))["defect"]]
    R = {(r["folder"], r["key"]): r for p in glob.glob(os.path.join(HERE, "gen_data", "rattle_part*.pkl")) for r in pickle.load(open(p, "rb"))}
    rows = []
    for d in D:
        if d["key"].split("-")[1] == d["key"].split("-")[2]:
            continue  # pristine
        f = os.path.join(sim, d["folder"])
        ref = yaml_pe(f"{f}/relaxed_{d['key']}.dump.yaml") - yaml_pe(f"{f}/unrelaxed_{d['key']}.dump.yaml")
        r = R.get((d["folder"], d["key"]))
        best = min([x["dE"] for x in r["runs"]]) if r else d["dE"]
        saddle = bool(r) and min(x["dE"] for x in r["runs"] if x["kind"] == "ref+rattle") < ref - 0.01
        rows.append((ref, best, saddle))
    ref, ours, sad = map(np.array, zip(*rows))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axs[0]
    ok = np.abs(ref) > 1e-3
    for m, c, lab in ((~sad, OI["blue"], "unique minimum"), (sad, OI["verm"], "LAMMPS label is a saddle point")):
        s = m & ok
        ax.scatter(-ref[s], -ours[s], s=18, color=c, label=f"{lab} ({s.sum()})", edgecolor="white", lw=0.3)
    lo, hi = 1e-2, 20
    ax.plot([lo, hi], [lo, hi], color=OI["black"], lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("team (LAMMPS) relaxation energy −ΔE (eV)"); ax.set_ylabel("ours (GRACE in ASE, + rattle) −ΔE (eV)")
    ax.set_title("a  relaxation energy, 120 pilot defect cells", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    ax = axs[1]
    diff = np.abs(ours - ref)[~sad]
    gain = (ref - ours)[sad]
    bins = np.logspace(-7, 0, 36)
    ax.hist(diff.clip(1e-7), bins=bins, color=OI["blue"], label="|ours − team|, unique minimum")
    ax.hist(gain, bins=bins, color=OI["verm"], label="energy gained below the team's saddle point")
    ax.set_xscale("log"); ax.set_xlabel("eV"); ax.set_ylabel("defect cells")
    ax.set_title("b  agreement", loc="left", fontweight="bold")
    ax.text(0.02, 0.97, f"median |difference| {1e3 * np.median(diff):.3f} meV\nbulk lattice constants within 3e-5 Å\n"
            f"displacements within 0.2% (median)", transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    fig.suptitle("Generating point-defect data ourselves: our pipeline vs the team's labels (30 compounds)", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_data_generation")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]
    for w in which:
        globals()[w]()
