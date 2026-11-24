"""Train a per-atom displacement model (where does each atom move during relaxation).

    python displacement/train_disp.py --domain point --model cgcnn \
        --view "rt=8.9,buf=4,edge=k3,feat=base+force" --head eqv --epochs 400

Split: by chemistry (element pair), 70/15/15, fixed by --split-seed. Every structure of
one compound is in the same part.

Metrics (Angstrom; computed on the atoms that carry the loss, i.e. within rt of the defect
for point defects, all atoms for planar):
  r_disp    PRIMARY. Per structure ||d_pred - d_true||_2 / ||d_true||_2 over its atoms,
            median over structures that move (excludes null 1-1-1 defects) and are stable.
            1.0 = predicting no motion; 0 = perfect.
  skill     1 - sum|err|^2 / sum|d|^2 pooled over stable moving structures.
  mae       mean |d_pred - d_true| per atom (A); zero_mae = mean |d_true| (baseline).
  cos       median cosine(pred, true) over atoms that move > 0.01 A.
  *_all     the same including structures flagged unstable (global rearrangements).
  cell_mae  planar: mean |error| of the predicted c-vector change (A).
Checkpoint: best validation r_disp. Test metrics are reported for that checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from atom_encodings import encode_dataset  # noqa: E402
from edge_features import encode_edges  # noqa: E402
from models import DisplacementModel, PaiNNDisplacement  # noqa: E402
from ext.wrapper import KINDS as EXT_KINDS, ExtRelax  # noqa: E402
from train_atom_encoding_sweep import WANDB_PROJECT_DEFAULT, WandbLogger, _git_commit  # noqa: E402
from views import make_view, parse_spec, spec_key  # noqa: E402

_SUPERSET = {}


BENCH_DIR = os.path.join(HERE, "bench_data", "views")


POINT_EXACT = False  # --e-exact: point labels from data/superset_point_exact.pt (exact LAMMPS totals)
MIXED = "off"  # --mixed: off | eval (mixed cells in val/test only) | train (also in training); dom = 2


def load_views(domain, spec):
    if domain.startswith("bench:"):  # external benchmark: prebuilt views (bench/build_views.py), spec ignored
        if domain not in _SUPERSET:
            _SUPERSET[domain] = torch.load(os.path.join(BENCH_DIR, domain[6:] + ".pt"), weights_only=False)
        return _SUPERSET[domain]  # (one load per process; modified in place)
    if domain == "joint":  # point defects + planar faults in one dataset; g.dom = 0 point / 1 planar
        out = []
        for k, d in enumerate(("point", "planar", "mixed")):
            if d == "mixed" and MIXED == "off":
                continue
            for v in load_views(d, spec):
                v.dom = torch.tensor([k]); out.append(v)
        return out
    if domain not in _SUPERSET:
        ddir = os.environ.get("DISP_DATA_DIR", os.path.join(HERE, "data"))
        _SUPERSET[domain] = torch.load(os.path.join(ddir, superset_file(domain)), weights_only=False)
    return [make_view(g, spec) for g in _SUPERSET[domain]]


def superset_file(domain):
    return "superset_point_exact.pt" if domain == "point" and POINT_EXACT else f"superset_{domain}.pt"


def union_keys():
    """Compounds of point AND planar data: one compound split shared by both domains (joint models and their
    separate-model baselines), so no compound is in training for one domain and in test for the other."""
    keys = set()
    for d in ("point", "planar"):
        if d not in _SUPERSET:
            ddir = os.environ.get("DISP_DATA_DIR", os.path.join(HERE, "data"))
            _SUPERSET[d] = torch.load(os.path.join(ddir, superset_file(d)), weights_only=False)
        keys |= {f"{g.element_a}-{g.element_b}" for g in _SUPERSET[d]}
    return keys


def split_by_compound(ds, seed, frac=(0.7, 0.15), keys=None):
    if hasattr(ds[0], "split"):  # benchmarks: the authors' fixed train/val/test split
        idx = [[], [], []]
        for i, g in enumerate(ds):
            idx[int(g.split)].append(i)
        return idx
    keys = sorted(keys or {f"{g.element_a}-{g.element_b}" for g in ds})
    rng = np.random.default_rng(1000 + seed); rng.shuffle(keys)
    n1 = int(round(frac[0] * len(keys))); n2 = n1 + int(round(frac[1] * len(keys)))
    part = {k: (0 if i < n1 else 1 if i < n2 else 2) for i, k in enumerate(keys)}
    idx = [[], [], []]
    for i, g in enumerate(ds):
        idx[part[f"{g.element_a}-{g.element_b}"]].append(i)
    return idx


def loss_fn(pred, y, kind):
    if kind == "mse":
        return ((pred - y) ** 2).sum(-1).mean()
    if kind == "huber":
        return F.smooth_l1_loss(pred, y, beta=1.0, reduction="none").sum(-1).mean()
    if kind == "l1":
        return (pred - y).norm(dim=-1).mean()
    raise ValueError(kind)


def energy_metrics(items):
    """Relaxation energy (sum over the graph's atoms of pe1 - pe0, eV) on stable structures: MAE of the total,
    median relative error |err| / |true| (|true| > 1 meV), skill 1 - sum err^2 / sum true^2, per-atom MAE."""
    st = [it for it in items if not it["unstable"]]
    if not st:
        return {}
    tp = np.array([float(it["e_pred"].sum()) for it in st]); tt = np.array([float(it["e_true"].sum()) for it in st])
    err = np.abs(tp - tt); big = np.abs(tt) > 1e-3
    return dict(e_tot_mae=float(err.mean()), e_tot_zero_mae=float(np.abs(tt).mean()),
                e_rel=float(np.median(err[big] / np.abs(tt[big]))) if big.any() else float("nan"),
                e_skill=float(1 - (err ** 2).sum() / max((tt ** 2).sum(), 1e-12)),
                e_atom_mae=float(torch.cat([(it["e_pred"] - it["e_true"]).abs() for it in st]).mean()))


def joint_metrics(items):
    """Per-domain metrics (point on the 8.9 A region, planar on all atoms) + their means for model selection."""
    mp = metrics([it for it in items if it["dom"] == 0], R_EVAL_POINT)
    ml = metrics([it for it in items if it["dom"] == 1])
    if items and "e_pred" in items[0]:
        mp.update(energy_metrics([it for it in items if it["dom"] == 0]))
        ml.update(energy_metrics([it for it in items if it["dom"] == 1]))
    res = {**{f"point_{k}": v for k, v in mp.items()}, **{f"planar_{k}": v for k, v in ml.items()}}
    mx = [it for it in items if it["dom"] == 2]
    if mx:  # mixed cells: all atoms; the point-defect region (8.9 A) of cells with a point defect; the pure faults
        mm = metrics(mx)
        mm.update({f"pd_{k}": v for k, v in metrics([it for it in mx if it["key"] != "none"], R_EVAL_POINT).items()})
        mm.update({f"none_{k}": v for k, v in metrics([it for it in mx if it["key"] == "none"]).items()})
        if "e_pred" in mx[0]:
            mm.update(energy_metrics(mx))
        res.update({f"mixed_{k}": v for k, v in mm.items()})
    for k in ("r_disp", "skill", "mae", "zero_mae", "e_rel"):
        if k in mp and k in ml:
            res[k] = 0.5 * (mp[k] + ml[k])
    if "e_rel" in res:
        res["mt"] = res["r_disp"] + res["e_rel"]  # multi-task model selection
    return res


@torch.no_grad()
def predict(model, loader, device, scale, cell_scale, planar, mode=None, e_harm=False):
    model.eval(); out = []
    for b in loader:
        b = b.to(device)
        d, cell = model(b)
        if mode == "joint":
            dg = joint_gauge(model, b, d)
            if getattr(model, "e_atom", None) is not None:
                ea_all = atom_energy(model, b, dg, e_harm, False)
            d = dg * b.dscale[b.batch].unsqueeze(-1)
        else:
            if planar and mode in (None, "planar"):  # translation gauge: targets have zero mean per structure
                d = d - scatter(d, b.batch, 0, reduce="mean")[b.batch]
            d = d * scale
        for k in range(b.num_graphs):
            m = (b.batch == k)
            item = dict(pred=d[m].cpu(), y=b.y[m].cpu(), mask=b.loss_mask[m].cpu(), dist=b.dist[m].cpu(),
                        unstable=bool(b.unstable[k]),
                        null=bool(b.null[k]), folder=b.folder[k], key=b.key[k])
            if mode in ("fixed", "cell"):
                item.update(pos0=b.pos0[m].cpu(), cell0=b.cell0[k].cpu(), cell_r=b.cell_r[k].cpu())
            if mode == "joint":
                item["dom"] = int(b.dom[k])
                if getattr(model, "e_atom", None) is not None:
                    item["e_pred"] = (ea_all[m] * b.edsc[k]).cpu(); item["e_true"] = b.de[m].cpu()
                    item["unstable"] = bool(b.unstable[k])
            if cell is not None and (mode != "joint" or item["dom"] >= 1):
                item["cell_pred"] = (cell[k] * cell_scale).cpu()
                item["cell_y"] = (b.cell_T[k] if mode == "cell" else b.cell_y[k]).cpu()
            out.append(item)
    return out


def joint_gauge(model, b, d):
    """Joint model: output in per-domain target units (ExtRelax returns A, model.scale = 1), and the
    planar translation gauge (zero mean per structure) on planar graphs only."""
    if isinstance(model, ExtRelax):
        d = d / b.dscale[b.batch].unsqueeze(-1)
    per = getattr(b, "pbc", b.dom)  # unified: translation gauge on periodic full cells (an input property)
    pl = (per[b.batch] == 1).unsqueeze(-1).float()
    return d - pl * scatter(d, b.batch, 0, reduce="mean")[b.batch]


DCAP = 12.9  # unified model: distance cap (A) = "no defect of this kind nearby" (= rt + buf of the point views)


def unify(ds):
    """Unified inputs: nothing tells the network which data set a graph comes from; point defects and planar
    faults get SEPARATE channels, so a cell can contain both.
      node x[:, 2:4] (is_defect, dist to defect) -> is_point_defect, dist_point (capped), is_fault_atom,
                                                     dist_fault (capped); cap = no such defect within DCAP
      edge col 2 (defect edge) -> point-defect edge, fault edge
      g.pbc = 1 for periodic full cells (translation gauge), 0 for point-defect cut-out clusters.
    Point data: fault channels empty. Planar data: is_defect marks the fault-layer atoms -> fault channels.
    Mixed cells (views with .is_fault / .dist_fault): is_defect / dist are the point defect's, plus the fault's."""
    for g in ds:
        x, e = g.x, g.edge_attr
        n = x.shape[0]; cap = torch.full((n,), DCAP); zero = torch.zeros(n)
        isd, dd = x[:, 2], x[:, 3].clamp(max=DCAP)
        if hasattr(g, "dist_fault"):
            isP, dP, isF, dF = isd, dd, g.is_fault.float(), g.dist_fault.clamp(max=DCAP)
            if not bool((isP > 0.5).any()):
                dP = cap
            if not bool((isF > 0.5).any()):
                dF = cap
            del g.is_fault, g.dist_fault  # consumed (only mixed views carry them; would break batching)
        elif int(g.dom) == 0:
            isP, dP, isF, dF = isd, dd, zero, cap
        else:
            isP, dP, isF, dF = zero, cap, isd, (dd if bool((isd > 0.5).any()) else cap)
        g.x = torch.cat([x[:, :2], torch.stack([isP, dP, isF, dF], -1), x[:, 4:]], -1)
        c, nb = g.edge_index[1], g.edge_index[0]
        eP = ((isP[c] > 0.5) | (isP[nb] > 0.5)).float(); eF = ((isF[c] > 0.5) | (isF[nb] > 0.5)).float()
        g.edge_attr = torch.cat([e[:, :2], eP.unsqueeze(-1), eF.unsqueeze(-1), e[:, 3:]], -1)
        g.pbc = torch.tensor([0 if (int(g.dom) == 0 and not getattr(g, "full_cell", False)) else 1])
    return ds


def atom_energy(model, b, d, e_harm, detach):
    """Per-atom relaxation energy in units of b.edsc: learned head (+ harmonic -1/2 F0 . d_pred, d in A)."""
    e = model.e_atom
    if e_harm:
        dA = d * b.dscale[b.batch].unsqueeze(-1)
        if detach:
            dA = dA.detach()
        e = e + (-0.5 * (b.force0 * dA).sum(-1)) / b.edsc[b.batch]
    return e


def joint_loss(d, b, kind, balance):
    """Position loss; balance = mean of the point and the planar loss (planar cells have ~12 atoms, point
    clusters hundreds, so a plain atom mean is dominated by point defects)."""
    m = b.loss_mask
    if not balance:
        return loss_fn(d[m], b.ys[m], kind)
    nd = b.dom[b.batch]
    parts = [loss_fn(d[m & (nd == k)], b.ys[m & (nd == k)], kind) for k in (0, 1, 2) if bool((m & (nd == k)).any())]
    return sum(parts) / len(parts)


def energy_loss(e, b, balance):
    """Relaxation-energy loss in per-domain units: per-atom L1 + |total error| / RMS total, per domain."""
    nd = b.dom[b.batch]
    tot_p = scatter(e, b.batch, 0, dim_size=b.num_graphs, reduce="sum")
    tot_t = scatter(b.es, b.batch, 0, dim_size=b.num_graphs, reduce="sum")
    ok = ~b.unstable.view(-1)
    parts = []
    for k in ((0, 1, 2) if balance else (None,)):
        an = (nd == k) if k is not None else torch.ones_like(nd, dtype=torch.bool)
        gn = ((b.dom == k) if k is not None else torch.ones_like(b.dom, dtype=torch.bool)) & ok
        an = an & ok[b.batch]
        if bool(gn.any()):
            w = b.eaw[b.batch][an] if hasattr(b, "eaw") else 1.0
            parts.append(((e[an] - b.es[an]).abs() * w).mean() + ((tot_p[gn] - tot_t[gn]).abs() / b.etsc[gn]).mean())
    return sum(parts) / len(parts) if parts else e.sum() * 0


R_EVAL_POINT = 8.9  # every point model is scored on the atoms within 8.9 A of the defect


def metrics(items, r_eval=None):
    """r_eval: score atoms with dist <= r_eval (predictions outside the model's own loss region
    count as zero motion). None = the loss region itself (planar: all atoms)."""
    if r_eval is not None:
        items = [dict(it, pred=torch.where(it["mask"].unsqueeze(-1), it["pred"], torch.zeros_like(it["pred"])),
                      mask=it["dist"] <= r_eval + 1e-6) for it in items]
    res = {}
    for tag, sel in (("", lambda it: not it["unstable"]), ("_all", lambda it: True)):
        r, num, den, errs, mags, coss = [], 0.0, 0.0, [], [], []
        for it in items:
            if not sel(it):
                continue
            p, y = it["pred"][it["mask"]], it["y"][it["mask"]]
            e = (p - y).norm(dim=-1); a = y.norm(dim=-1)
            errs.append(e); mags.append(a)
            mv = a > 0.01
            if mv.any():
                coss.append(F.cosine_similarity(p[mv], y[mv], dim=-1))
            if it["null"]:
                continue
            r.append(float(e.pow(2).sum().sqrt() / a.pow(2).sum().sqrt().clamp(min=1e-12)))
            num += float(e.pow(2).sum()); den += float(a.pow(2).sum())
        if not errs:
            for k in ("r_disp", "r_disp_mean", "skill", "mae", "zero_mae", "cos"):
                res[f"{k}{tag}"] = float("nan")
            continue
        errs = torch.cat(errs); mags = torch.cat(mags)
        res[f"r_disp{tag}"] = float(np.median(r)) if r else float("nan")
        res[f"r_disp_mean{tag}"] = float(np.mean(r)) if r else float("nan")
        res[f"skill{tag}"] = 1 - num / den if den > 0 else float("nan")
        res[f"mae{tag}"] = float(errs.mean()); res[f"zero_mae{tag}"] = float(mags.mean())
        res[f"cos{tag}"] = float(torch.cat(coss).median()) if coss else float("nan")
    null = [it for it in items if it["null"]]
    if null:
        res["null_max_pred"] = float(max(it["pred"][it["mask"]].norm(dim=-1).max() for it in null))
    if items and "cell_pred" in items[0] and any(not it["unstable"] for it in items):
        st = [it for it in items if not it["unstable"]]
        res["cell_mae"] = float(torch.stack([(it["cell_pred"] - it["cell_y"]).abs().mean() for it in st]).mean())
        res["cell_zero_mae"] = float(torch.stack([it["cell_y"].abs().mean() for it in st]).mean())
    if items and "pos0" in items[0]:
        res.update(bench_metrics(items))
    return res


@torch.no_grad()
def bench_metrics(items):
    """The authors' scores (mean over test structures): mae_pos = mean |x_pred - x_relaxed| per Cartesian
    component over real atoms (minimum image in the relaxed cell, as E3Relax-H2; DefiNet: no vacancy
    dummies); cell: Frobenius error of the metric tensor C C^T and |volume error|. *_dummy = unrelaxed input."""
    mp, md, mt, mtd, mv, mvd = [], [], [], [], [], []
    for it in items:
        c0, cr = it["cell0"].double(), it["cell_r"].double()
        cp = c0 @ (torch.eye(3, dtype=c0.dtype) + it["cell_pred"].double()) if "cell_pred" in it and it["cell_pred"].shape == (3, 3) else c0
        s0 = it["pos0"].double() @ torch.linalg.inv(c0)
        s_true = s0 + it["y"].double() @ torch.linalg.inv(c0)
        s_pred = s0 + it["pred"].double() @ torch.linalg.inv(c0)
        x_true = s_true @ cr
        m = it["mask"]

        def mae(x):
            df = (x - x_true) @ torch.linalg.inv(cr)
            return float(((df - torch.round(df)) @ cr)[m].abs().mean())
        mp.append(mae(s_pred @ cp)); md.append(mae(s0 @ c0))
        G = lambda c: c @ c.T  # noqa: E731
        mt.append(float(torch.linalg.norm(G(cp) - G(cr)))); mtd.append(float(torch.linalg.norm(G(c0) - G(cr))))
        mv.append(float((torch.det(cp) - torch.det(cr)).abs())); mvd.append(float((torch.det(c0) - torch.det(cr)).abs()))
    return dict(mae_pos=float(np.mean(mp)), mae_pos_dummy=float(np.mean(md)), mae_metric=float(np.mean(mt)),
                mae_metric_dummy=float(np.mean(mtd)), mae_vol=float(np.mean(mv)), mae_vol_dummy=float(np.mean(mvd)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="point | planar | point_full | bench:<name> (bench_data/views)")
    ap.add_argument("--zemb", action="store_true", help="PaiNN: learned element embedding")
    ap.add_argument("--bench-feats", default="ours", choices=["ours", "none"],
                    help="benchmarks: 'none' = no GRACE inputs (x = 1, edges = distance/same-element/defect, F0 = 0)")
    ap.add_argument("--model", choices=["cgcnn", "transformer", "painn", "definet", "e3relax", "e3relaxh2"], required=True)
    ap.add_argument("--view", default="")
    ap.add_argument("--head", default="eqv", choices=["eqv", "eqvF", "direct"])
    ap.add_argument("--head-force", action="store_true", help="give F0.u projections to the head MLP")
    ap.add_argument("--loss", default="mse", choices=["mse", "huber", "l1"])
    ap.add_argument("--encoding", default="pair_props_edge")
    ap.add_argument("--edge-enc", default=None, help="rbf16 (CGCNN default) | raw (Transformer default)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--lambda-cell", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--union-split", action="store_true",
                    help="split over the compounds of point AND planar data (always on for --domain joint)")
    ap.add_argument("--no-domflag", action="store_true", help="joint: no point/planar one-hot in the node features")
    ap.add_argument("--energy-weight", type=float, default=0.0,
                    help="joint PaiNN: weight of the relaxation-energy loss (per-atom pe1-pe0 + graph total); 0 = off")
    ap.add_argument("--pos-weight", type=float, default=1.0, help="weight of the position (and cell) loss; 0 = energy only")
    ap.add_argument("--e-coupling", action="store_true", help="energy head + a * F0.d_pred (harmonic coupling term)")
    ap.add_argument("--unified", action="store_true",
                    help="joint without any domain input: no domain flag, one displacement / energy scale, distance-to-"
                         "point-defect capped (planar = cap), gauge on periodic cells; for cells with point + planar defects")
    ap.add_argument("--mixed", default="off", choices=["off", "eval", "train"],
                    help="unified: mixed cells (point defect in a planar-fault cell, data/superset_mixed.pt) in "
                         "val/test only (eval) or also in training (train); split by compound like the rest")
    ap.add_argument("--e-exact", action="store_true",
                    help="point labels with exact LAMMPS energy totals + 8 far-field cells flagged (data/superset_point_exact.pt)")
    ap.add_argument("--e-harm", action="store_true", help="energy = head + harmonic -1/2 F0 . d_pred (head learns the residual)")
    ap.add_argument("--e-harm-detach", action="store_true", help="--e-harm without energy gradients into the displacements")
    ap.add_argument("--final", action="store_true",
                    help="production training: 90/10 train/val over all compounds (union), no test set (test := val)")
    ap.add_argument("--no-dom-balance", action="store_true", help="joint: plain atom-mean loss instead of point/planar mean")
    ap.add_argument("--select", default=None,
                    help="validation metric for the checkpoint (default r_disp; mae_pos for bench:*, the authors' "
                         "coordinate MAE, which also scores the predicted cell)")
    ap.add_argument("--group", default="disp-screen")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--norm", action="store_true", help="LayerNorm in the CGCNN stack and before the head (PaiNN: scalar LN + vector RMS norm)")
    ap.add_argument("--ext-feats", action="store_true", help="DefiNet/E3Relax: add our node features to the atom embedding")
    ap.add_argument("--ext-force", action="store_true", help="DefiNet/E3Relax: initial force as start vector")
    ap.add_argument("--layerwise", type=int, default=None,
                    help="loss on every layer's positions (default: 1 for e3relax/e3relaxh2, 0 for definet)")
    ap.add_argument("--lambda-ext-cell", type=float, default=1.0, help="weight of the per-layer lattice loss (E3Relax)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights used for evaluation (0 = off)")
    ap.add_argument("--amsgrad", action="store_true")
    ap.add_argument("--ext-static", action="store_true", help="DefiNet ablation: edge geometry from the initial positions in every layer")
    ap.add_argument("--ext-novn", action="store_true", help="DefiNet ablation: no virtual global node")
    ap.add_argument("--std-inputs", action="store_true",
                    help="z-score node features and edge features (except col 0 = distance) with train statistics")
    a = ap.parse_args()
    edge_enc = a.edge_enc or ("raw" if a.model == "transformer" else "rbf16")
    ext = a.model in EXT_KINDS
    if ext and a.layerwise is None:
        a.layerwise = 0 if a.model == "definet" else 1
    spec = parse_spec(a.view)
    if a.unified:
        a.domain, a.no_domflag = "joint", True
    if a.mixed != "off" and not a.unified:
        raise SystemExit("--mixed needs --unified")
    global MIXED, POINT_EXACT
    MIXED = a.mixed; POINT_EXACT = a.e_exact
    if a.e_harm_detach:
        a.e_harm = True
    joint = a.domain == "joint"
    planar = a.domain == "planar" or joint  # joint: the model gets the planar cell head
    bench = a.domain.startswith("bench:")
    r_eval = None if (planar or bench) else R_EVAL_POINT
    union = a.union_split or joint
    run_key = f"{a.domain.replace(':', '-')}_{a.model}_{spec_key(spec)}_{a.head}{'H' if a.head_force else ''}_{a.loss}" \
              f"{'_norm' if a.norm else ''}{'_std' if a.std_inputs else ''}" \
              f"{('_xf' if a.ext_feats else '') + ('_xF' if a.ext_force else '') + ('_lw' if a.layerwise else '') + (f'_h{a.hidden}') if ext else ''}" \
              f"{'_ema' if a.ema else ''}{'_zemb' if a.zemb else ''}{'_nofeat' if bench and a.bench_feats == 'none' else ''}" \
              f"{'_nodf' if joint and a.no_domflag else ''}{'_nobal' if joint and a.no_dom_balance else ''}" \
              f"{'_us' if a.union_split and not joint else ''}" \
              f"{f'_ew{a.energy_weight:g}' if a.energy_weight else ''}{'_pw0' if a.pos_weight == 0 else ''}" \
              f"{'_ec' if a.e_coupling else ''}{'_uni' if a.unified else ''}{'_mx' if a.mixed == 'train' else ''}{('_ehd' if a.e_harm_detach else '_eh') if a.e_harm else ''}{'_ex' if a.e_exact else ''}{'_final' if a.final else ''}{'_' + a.tag if a.tag else ''}_s{a.seed}_sp{a.split_seed}"
    os.makedirs(os.path.join(a.out_dir, "results"), exist_ok=True)
    res_path = os.path.join(a.out_dir, "results", f"{run_key}.json")
    if os.path.isfile(res_path):
        print(f"exists: {res_path}"); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ds = load_views(a.domain, spec)
    mode = ds[0].mode if bench else ("joint" if joint else "planar" if planar else "point")
    if a.unified:
        ds = unify(ds)
    if bench and a.bench_feats == "none":
        for g in ds:
            g.x = torch.ones(g.num_nodes, 1); g.edge_attr = g.edge_attr[:, :3].clone(); g.force0 = torch.zeros_like(g.force0)
            g.stress0 = torch.zeros_like(g.stress0)
    planar = planar or mode == "cell"  # cell head + cell scale; translation gauge stays planar-only
    if bench:  # multi-element: own-element descriptors + element-aware edge features (as pair_props_edge)
        from atom_encodings import element_table
        tab = element_table()
        for g in ds:
            z = g.z.long(); nb_, ct_ = g.edge_index
            g.x = torch.cat([tab.desc[z], g.x], -1)
            rsum = tab.radius[z[nb_]] + tab.radius[z[ct_]]
            g.edge_attr = torch.cat([g.edge_attr, (g.edge_attr[:, 0] / rsum.clamp(min=1e-6)).unsqueeze(-1),
                                     (tab.en[z[nb_]] - tab.en[z[ct_]]).abs().unsqueeze(-1)], -1)
    else:
        ds = encode_dataset(ds, a.encoding)
    if joint and not a.no_domflag:
        for g in ds:
            g.x = torch.cat([g.x, F.one_hot(g.dom.expand(g.num_nodes), 2).float()], -1)
    ds = encode_edges(ds, edge_enc)
    tr_i, va_i, te_i = split_by_compound(ds, a.split_seed, frac=(0.9, 0.1) if a.final else (0.7, 0.15),
                                         keys=union_keys() if (union or a.final) else None)
    if a.final:
        te_i = list(va_i)  # no test set in production training; "test" metrics = validation
    if a.mixed == "eval":
        tr_i = [i for i in tr_i if int(ds[i].dom) != 2]
    energy = a.energy_weight > 0
    if energy and not (joint and a.model == "painn"):
        raise SystemExit("--energy-weight: joint PaiNN only")
    train = [ds[i] for i in tr_i if not (spec["stable"] and bool(ds[i].unstable))]
    # target scale: RMS displacement component over training loss atoms of stable structures
    ys = torch.cat([g.y[g.loss_mask] for g in train if not bool(g.unstable)])
    scale = float(ys.pow(2).mean().sqrt())
    cs = torch.cat([(g.cell_T if mode == "cell" else g.cell_y) for g in train if not bool(g.unstable)
                    and (not joint or int(g.dom) >= 1)])
    cell_scale = float(cs.pow(2).mean().sqrt()) if planar else 1.0
    dscales = None
    if joint:  # one target scale per domain (planar displacements are much smaller)
        dscales = [float(torch.cat([g.y[g.loss_mask] for g in train if not bool(g.unstable) and int(g.dom) == k])
                         .pow(2).mean().sqrt()) for k in (0, 1)]
        if a.unified:  # one displacement unit for every structure (point / planar statistics, as without mixed)
            dscales = [float(np.sqrt(np.mean([s ** 2 for s in dscales])))] * 3
        for g in ds:
            g.dscale = torch.tensor([dscales[int(g.dom)]])
    escales = None; EA = None
    if energy:  # per-domain RMS of the per-atom and of the total relaxation energy (train, stable)
        escales = []
        for k in (0, 1, 2):
            tg = [g for g in train if not bool(g.unstable) and int(g.dom) == k] or \
                 [g for g in train if not bool(g.unstable) and int(g.dom) == 1]  # mixed eval-only: planar stats
            ea = float(torch.cat([g.de for g in tg]).pow(2).mean().sqrt())
            et = float(torch.tensor([float(g.de.sum()) for g in tg]).pow(2).mean().sqrt())
            escales.append((ea, et))
        EA = float(np.sqrt(np.mean([e[0] ** 2 for e in escales[:2]])))  # unified: one global energy unit
        for g in ds:
            ea, et = escales[int(g.dom)]
            if a.unified:  # output in EA units for every graph; per-source normalisation only inside the loss
                g.es = g.de / EA; g.edsc = torch.tensor([EA]); g.etsc = torch.tensor([et / EA])
                g.eaw = torch.tensor([EA / ea])
            else:
                g.es = g.de / ea; g.edsc = torch.tensor([ea]); g.etsc = torch.tensor([et / ea])
    for g in ds:
        g.ys = g.y / (float(g.dscale) if joint else scale); g.cs = (g.cell_T if mode == "cell" else g.cell_y) / cell_scale
    norm_stats = None
    if a.std_inputs:
        X = torch.cat([g.x for g in train]); E = torch.cat([g.edge_attr for g in train])
        xm, xs = X.mean(0), X.std(0).clamp(min=1e-6)
        em, es = E.mean(0), E.std(0).clamp(min=1e-6)
        em[0], es[0] = 0.0, 1.0  # keep the raw distance (used by the vector head)
        for g in ds:
            g.x = (g.x - xm) / xs; g.edge_attr = (g.edge_attr - em) / es
        norm_stats = dict(xm=xm, xs=xs, em=em, es=es)
    print(f"[{run_key}] graphs {len(ds)} train {len(train)} val {len(va_i)} test {len(te_i)} | "
          f"x {ds[0].x.shape[1]} e {ds[0].edge_attr.shape[1]} | scale {scale:.4f} A | "
          f"{f'dscales {dscales[0]:.4f}/{dscales[1]:.4f} A | ' if joint else ''}nodes/graph {np.mean([g.num_nodes for g in ds]):.0f} edges/graph {np.mean([g.edge_index.shape[1] for g in ds]):.0f} "
          f"| load {time.time() - t0:.0f}s", flush=True)
    tl = DataLoader(train, batch_size=a.bs, shuffle=True)
    vl = DataLoader([ds[i] for i in va_i], batch_size=a.bs)
    sl = DataLoader([ds[i] for i in te_i], batch_size=a.bs)
    if ext:
        model = ExtRelax(a.model, feat_dim=ds[0].x.shape[1] if a.ext_feats else None, force_init=a.ext_force,
                         hidden=a.hidden, num_layers=a.layers, planar=planar and not joint,
                         static_geom=a.ext_static, no_vn=a.ext_novn, mode="fixed" if joint else mode).to(device)
        model.set_scales(1.0 if joint else scale, cell_scale)
    elif a.model == "painn":  # head eqvF = vector features initialised with the initial force
        model = PaiNNDisplacement(ds[0].x.shape[1], ds[0].edge_attr.shape[1], hidden=a.hidden, num_layers=a.layers,
                                  force_init=a.head == "eqvF", cell_head="tensor" if mode == "cell" else planar,
                                  norm=a.norm, zemb=a.zemb, energy_head=energy, e_coupling=a.e_coupling).to(device)
    else:
        model = DisplacementModel(a.model, ds[0].x.shape[1], ds[0].edge_attr.shape[1], head=a.head, hidden=a.hidden,
                                  num_layers=a.layers, use_force_in_head=a.head_force, cell_head=planar,
                                  norm=a.norm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, amsgrad=a.amsgrad)
    ema = None
    if a.ema:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)
    evm = lambda: ema.module if ema is not None else model  # noqa: E731
    steps = a.epochs * len(tl); warm = a.warmup * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * (
        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(steps, 1))))))
    cfg = dict(vars(a), edge_enc=edge_enc, spec=spec, run_key=run_key, n_params=n_params, scale_A=scale,
               cell_scale_A=cell_scale, dscales_A=dscales, escales_eV=escales, n_train=len(train), n_val=len(va_i), n_test=len(te_i),
               dataset=f"displacement/data/superset_{a.domain}.pt" + (" (point: superset_point_exact.pt)" if a.e_exact else ""), git=_git_commit(),
               slurm_job=os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
               slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID"), optimizer="AdamW",
               scheduler="warmup+cosine(min 1%)", grad_clip=1.0, split="by compound 70/15/15")
    wb = WandbLogger(enabled=not a.no_wandb, project=WANDB_PROJECT_DEFAULT, name=f"disp-{run_key}", config=cfg,
                     group=a.group, job_type="train",
                     tags=[a.domain, a.model, a.head, a.loss, spec["edge"], spec["feat"]])
    best, best_ep, best_state, curve = float("inf"), 0, None, []
    sel_key = a.select or ("mae_pos" if a.domain.startswith("bench:") else
                           ("e_rel" if a.pos_weight == 0 else "mt") if energy else "r_disp")
    n_skip = 0
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0; n = 0
        for b in tl:
            b = b.to(device)
            d, cell = model(b)
            if ext:
                # authors' scheme: position loss on the last (DefiNet) or every layer (E3Relax),
                # plus |cell - relaxed cell| per layer for the lattice-updating models
                sel = model.layers if a.layerwise else model.layers[-1:]
                if joint:
                    loss = sum(joint_loss(joint_gauge(model, b, dl), b, a.loss, not a.no_dom_balance) for dl, _ in sel)
                else:
                    loss = sum(loss_fn(dl[b.loss_mask], b.ys[b.loss_mask], a.loss) for dl, _ in sel)
                if a.model != "definet":
                    ct = model.cell_target(b) / model.scale
                    loss = loss + a.lambda_ext_cell * sum((cl - ct).abs().mean() for _, cl in sel)
            elif joint:
                dg = joint_gauge(model, b, d)
                loss = joint_loss(dg, b, a.loss, not a.no_dom_balance)
                plg = b.dom >= 1  # planar faults and mixed cells: the c vector relaxed
                if cell is not None and bool(plg.any()):
                    loss = loss + a.lambda_cell * F.smooth_l1_loss(cell[plg], b.cs[plg], beta=1.0)
                loss = a.pos_weight * loss
                if energy:
                    loss = loss + a.energy_weight * energy_loss(atom_energy(model, b, dg, a.e_harm, a.e_harm_detach), b,
                                                               not a.no_dom_balance)
            else:
                if mode == "planar":
                    d = d - scatter(d, b.batch, 0, reduce="mean")[b.batch]
                loss = loss_fn(d[b.loss_mask], b.ys[b.loss_mask], a.loss)
                if cell is not None:
                    loss = loss + a.lambda_cell * F.smooth_l1_loss(cell, b.cs, beta=1.0)
            opt.zero_grad()
            if not torch.isfinite(loss):  # skip a non-finite batch instead of poisoning the weights
                n_skip += 1; sched.step(); continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                n_skip += 1; opt.zero_grad(); sched.step(); continue
            opt.step(); sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tot += float(loss.detach()) * b.num_graphs; n += b.num_graphs
        if ep % a.eval_every == 0 or ep == a.epochs:
            vi = predict(evm(), vl, device, scale, cell_scale, planar, mode, a.e_harm)
            vm = joint_metrics(vi) if joint else metrics(vi, r_eval)
            curve.append(dict(epoch=ep, train_loss=tot / max(n, 1), **{f"val_{k}": v for k, v in vm.items()}))
            wb.log({"epoch": ep, "train/loss": tot / max(n, 1), **{f"val/{k}": v for k, v in vm.items()}}, step=ep)
            sv = vm.get(sel_key, float("nan"))
            if (np.isfinite(sv) and sv < best) or best_state is None:
                best, best_ep = (sv if np.isfinite(sv) else float("inf")), ep
                best_state = {k: v.detach().clone() for k, v in evm().state_dict().items()}
            if ep % (a.eval_every * 20) == 0:
                print(f"  ep {ep} loss {tot / max(n, 1):.4f} val r_disp {vm['r_disp']:.3f} skill {vm['skill']:.3f} "
                      f"mae {vm['mae']:.4f} (zero {vm['zero_mae']:.4f}) best {sel_key} {best:.4f}@{best_ep} skipped {n_skip} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    model.load_state_dict(best_state)  # (the EMA weights when --ema is set)
    val_items = predict(model, vl, device, scale, cell_scale, planar, mode, a.e_harm)
    test_items = predict(model, sl, device, scale, cell_scale, planar, mode, a.e_harm)
    if joint:
        vm, tm = joint_metrics(val_items), joint_metrics(test_items)
    else:
        vm, tm = metrics(val_items, r_eval), metrics(test_items, r_eval)
    if r_eval is not None and not joint:  # also on the model's own loss region
        vm.update({f"own_{k}": v for k, v in metrics(val_items).items()})
        tm.update({f"own_{k}": v for k, v in metrics(test_items).items()})
    out = dict(run_key=run_key, config=cfg, select=sel_key, n_skipped=n_skip, best_epoch=best_ep, val=vm, test=tm, curve=curve,
               train_time_s=time.time() - t0)
    json.dump(out, open(res_path, "w"), indent=1)
    if a.save_preds:
        os.makedirs(os.path.join(a.out_dir, "preds"), exist_ok=True)
        torch.save(dict(val=val_items, test=test_items, scale=scale, state=best_state, config=cfg,
                        norm=norm_stats, cell_scale=cell_scale, dscales=dscales, escales=escales,
                        x_dim=ds[0].x.shape[1], e_dim=ds[0].edge_attr.shape[1], e_unit=EA),
                   os.path.join(a.out_dir, "preds", f"{run_key}.pt"))
    wb.summary({**{f"test/{k}": v for k, v in tm.items()}, **{f"best_val/{k}": v for k, v in vm.items()},
                "best_epoch": best_ep, "n_params": n_params})
    wb.finish()
    print(f"[{run_key}] done best_ep {best_ep} | val r_disp {vm['r_disp']:.3f} skill {vm['skill']:.3f} | "
          f"test r_disp {tm['r_disp']:.3f} skill {tm['skill']:.3f} mae {tm['mae']:.4f} (zero {tm['zero_mae']:.4f}) "
          f"| {time.time() - t0:.0f}s | peak GPU mem "
          f"{torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
