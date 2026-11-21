"""Final-model plots: seed-ensemble test predictions pooled over the 3 compound splits.
    python displacement/plot_final.py
"""
import glob, os, re, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from train_disp import R_EVAL_POINT  # noqa: E402

FINAL = {"point": ("runs_p4", "point_full_painn_rt8.9_buf4_k3_hk3_base+force_st1_eqv_mse_norm_std"),
         "planar": ("runs_p5", "planar_painn_rt8.9_buf4_k3_hk3_base+force_st1_eqvF_mse_norm_std_lr5e-4")}

def ens_items(d, key):
    out = []
    for sp in (0, 1, 2):
        ps = sorted(glob.glob(os.path.join(HERE, d, "preds", f"{key}_s*_sp{sp}.pt")))
        P = [torch.load(p, weights_only=False)["test"] for p in ps]
        for its in zip(*P):
            it = dict(its[0]); it["pred"] = torch.stack([i["pred"] for i in its]).mean(0); out.append(it)
    return out

fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
for row, dom in enumerate(("point", "planar")):
    items = [i for i in ens_items(*FINAL[dom]) if not i["unstable"]]
    yt, yp, dist, rs = [], [], [], []
    for it in items:
        m = it["mask"].clone()
        if dom == "point": m = it["dist"] <= R_EVAL_POINT
        y, p = it["y"][m], it["pred"][m]
        yt.append(y); yp.append(p)
        if dom == "point": dist.append(it["dist"][m])
        if y.norm() > 1e-4: rs.append(float((p - y).norm() / y.norm()))
    yt, yp = torch.cat(yt).numpy(), torch.cat(yp).numpy()
    a = axs[row, 0]
    a.hexbin(yt.ravel(), yp.ravel(), gridsize=80, bins="log", cmap="Blues", mincnt=1, extent=(-0.6, 0.6, -0.6, 0.6))
    a.plot([-0.6, 0.6], [-0.6, 0.6], "k--", lw=.8); a.set_xlim(-.6, .6); a.set_ylim(-.6, .6)
    a.set_xlabel("true displacement component (Å)"); a.set_ylabel("predicted (Å)")
    r2 = 1 - ((yp - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    a.set_title(f"{dom}: per-atom components (test, pooled R² = {r2:.2f})")
    a = axs[row, 1]
    nt, npd = np.linalg.norm(yt, axis=1), np.linalg.norm(yp, axis=1)
    sel = nt > 1e-3
    a.loglog(nt[sel], npd[sel], ".", ms=1.5, alpha=.3, color="#2a6fdb"); a.plot([1e-3, 3], [1e-3, 3], "k--", lw=.8)
    a.set_xlim(1e-3, 3); a.set_ylim(1e-4, 3)
    cos = (yt * yp).sum(1)[sel] / (nt[sel] * np.maximum(npd[sel], 1e-12))
    a.set_xlabel("|true displacement| (Å)"); a.set_ylabel("|predicted| (Å)")
    a.set_title(f"magnitude (atoms with |d| > 1 mÅ; median cos = {np.median(cos):.3f})")
    a = axs[row, 2]
    if dom == "point":
        dist = torch.cat(dist).numpy(); err = np.linalg.norm(yp - yt, axis=1)
        bins = np.arange(0, 9.5, 0.75); c = 0.5 * (bins[1:] + bins[:-1])
        mt = [nt[(dist >= l) & (dist < h)].mean() for l, h in zip(bins[:-1], bins[1:])]
        me = [err[(dist >= l) & (dist < h)].mean() for l, h in zip(bins[:-1], bins[1:])]
        a.semilogy(c, mt, "o-", color="0.4", label="mean |true d|"); a.semilogy(c, me, "s-", color="#2a6fdb", label="mean |error|")
        a.set_xlabel("distance from defect (Å)"); a.set_ylabel("Å"); a.legend(); a.set_title("error vs distance from the defect")
    else:
        a.hist(rs, bins=np.linspace(0, 1.5, 41), color="#2a6fdb")
        a.axvline(np.median(rs), color="k", ls="--", lw=1)
        a.set_xlabel("per-structure r = |err| / |true|"); a.set_ylabel("structures")
        a.set_title(f"per-structure relative error (median {np.median(rs):.3f})")
    print(dom, "n structures", len(items), "median r", np.median(rs), "R2", r2)
fig.suptitle("Final displacement models (PaiNN, 2-seed ensemble), test compounds of 3 splits, stable structures")
fig.tight_layout(); out = os.path.join(HERE, "final_models.png"); fig.savefig(out, dpi=120); print(out)
