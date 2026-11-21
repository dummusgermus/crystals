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


def load_views(domain, spec):
    if domain not in _SUPERSET:
        ddir = os.environ.get("DISP_DATA_DIR", os.path.join(HERE, "data"))
        _SUPERSET[domain] = torch.load(os.path.join(ddir, f"superset_{domain}.pt"), weights_only=False)
    return [make_view(g, spec) for g in _SUPERSET[domain]]


def split_by_compound(ds, seed, frac=(0.7, 0.15)):
    keys = sorted({f"{g.element_a}-{g.element_b}" for g in ds})
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


@torch.no_grad()
def predict(model, loader, device, scale, cell_scale, planar):
    model.eval(); out = []
    for b in loader:
        b = b.to(device)
        d, cell = model(b)
        if planar:  # translation gauge: targets have zero mean per structure
            d = d - scatter(d, b.batch, 0, reduce="mean")[b.batch]
        d = d * scale
        for k in range(b.num_graphs):
            m = (b.batch == k)
            item = dict(pred=d[m].cpu(), y=b.y[m].cpu(), mask=b.loss_mask[m].cpu(), dist=b.dist[m].cpu(),
                        unstable=bool(b.unstable[k]),
                        null=bool(b.null[k]), folder=b.folder[k], key=b.key[k])
            if cell is not None:
                item["cell_pred"] = (cell[k] * cell_scale).cpu(); item["cell_y"] = b.cell_y[k].cpu()
            out.append(item)
    return out


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
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["point", "planar", "point_full"], required=True)
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
    planar = a.domain == "planar"
    r_eval = None if planar else R_EVAL_POINT
    run_key = f"{a.domain}_{a.model}_{spec_key(spec)}_{a.head}{'H' if a.head_force else ''}_{a.loss}" \
              f"{'_norm' if a.norm else ''}{'_std' if a.std_inputs else ''}" \
              f"{('_xf' if a.ext_feats else '') + ('_xF' if a.ext_force else '') + ('_lw' if a.layerwise else '') + (f'_h{a.hidden}') if ext else ''}" \
              f"{'_ema' if a.ema else ''}" \
              f"{'_' + a.tag if a.tag else ''}_s{a.seed}_sp{a.split_seed}"
    os.makedirs(os.path.join(a.out_dir, "results"), exist_ok=True)
    res_path = os.path.join(a.out_dir, "results", f"{run_key}.json")
    if os.path.isfile(res_path):
        print(f"exists: {res_path}"); return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ds = load_views(a.domain, spec)
    ds = encode_dataset(ds, a.encoding)
    ds = encode_edges(ds, edge_enc)
    tr_i, va_i, te_i = split_by_compound(ds, a.split_seed)
    train = [ds[i] for i in tr_i if not (spec["stable"] and bool(ds[i].unstable))]
    # target scale: RMS displacement component over training loss atoms of stable structures
    ys = torch.cat([g.y[g.loss_mask] for g in train if not bool(g.unstable)])
    scale = float(ys.pow(2).mean().sqrt())
    cs = torch.cat([g.cell_y for g in train if not bool(g.unstable)])
    cell_scale = float(cs.pow(2).mean().sqrt()) if planar else 1.0
    for g in ds:
        g.ys = g.y / scale; g.cs = g.cell_y / cell_scale
    if a.std_inputs:
        X = torch.cat([g.x for g in train]); E = torch.cat([g.edge_attr for g in train])
        xm, xs = X.mean(0), X.std(0).clamp(min=1e-6)
        em, es = E.mean(0), E.std(0).clamp(min=1e-6)
        em[0], es[0] = 0.0, 1.0  # keep the raw distance (used by the vector head)
        for g in ds:
            g.x = (g.x - xm) / xs; g.edge_attr = (g.edge_attr - em) / es
    print(f"[{run_key}] graphs {len(ds)} train {len(train)} val {len(va_i)} test {len(te_i)} | "
          f"x {ds[0].x.shape[1]} e {ds[0].edge_attr.shape[1]} | scale {scale:.4f} A | "
          f"nodes/graph {np.mean([g.num_nodes for g in ds]):.0f} edges/graph {np.mean([g.edge_index.shape[1] for g in ds]):.0f} "
          f"| load {time.time() - t0:.0f}s", flush=True)
    tl = DataLoader(train, batch_size=a.bs, shuffle=True)
    vl = DataLoader([ds[i] for i in va_i], batch_size=a.bs)
    sl = DataLoader([ds[i] for i in te_i], batch_size=a.bs)
    if ext:
        model = ExtRelax(a.model, feat_dim=ds[0].x.shape[1] if a.ext_feats else None, force_init=a.ext_force,
                         hidden=a.hidden, num_layers=a.layers, planar=planar,
                         static_geom=a.ext_static, no_vn=a.ext_novn).to(device)
        model.set_scales(scale, cell_scale)
    elif a.model == "painn":  # head eqvF = vector features initialised with the initial force
        model = PaiNNDisplacement(ds[0].x.shape[1], ds[0].edge_attr.shape[1], hidden=a.hidden, num_layers=a.layers,
                                  force_init=a.head == "eqvF", cell_head=planar, norm=a.norm).to(device)
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
               cell_scale_A=cell_scale, n_train=len(train), n_val=len(va_i), n_test=len(te_i),
               dataset=f"displacement/data/superset_{a.domain}.pt", git=_git_commit(),
               slurm_job=os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
               slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID"), optimizer="AdamW",
               scheduler="warmup+cosine(min 1%)", grad_clip=1.0, split="by compound 70/15/15")
    wb = WandbLogger(enabled=not a.no_wandb, project=WANDB_PROJECT_DEFAULT, name=f"disp-{run_key}", config=cfg,
                     group=a.group, job_type="train",
                     tags=[a.domain, a.model, a.head, a.loss, spec["edge"], spec["feat"]])
    best, best_ep, best_state, curve = float("inf"), 0, None, []
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0; n = 0
        for b in tl:
            b = b.to(device)
            d, cell = model(b)
            if ext:
                # authors' scheme: position loss on the last (DefiNet) or every layer (E3Relax),
                # plus |cell - relaxed cell| per layer for the lattice-updating models
                sel = model.layers if a.layerwise else model.layers[-1:]
                loss = sum(loss_fn(dl[b.loss_mask], b.ys[b.loss_mask], a.loss) for dl, _ in sel)
                if a.model != "definet":
                    ct = model.cell_target(b) / model.scale
                    loss = loss + a.lambda_ext_cell * sum((cl - ct).abs().mean() for _, cl in sel)
            else:
                if planar:
                    d = d - scatter(d, b.batch, 0, reduce="mean")[b.batch]
                loss = loss_fn(d[b.loss_mask], b.ys[b.loss_mask], a.loss)
                if cell is not None:
                    loss = loss + a.lambda_cell * F.smooth_l1_loss(cell, b.cs, beta=1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tot += float(loss.detach()) * b.num_graphs; n += b.num_graphs
        if ep % a.eval_every == 0 or ep == a.epochs:
            vm = metrics(predict(evm(), vl, device, scale, cell_scale, planar), r_eval)
            curve.append(dict(epoch=ep, train_loss=tot / n, **{f"val_{k}": v for k, v in vm.items()}))
            wb.log({"epoch": ep, "train/loss": tot / n, **{f"val/{k}": v for k, v in vm.items()}}, step=ep)
            if vm["r_disp"] < best or best_state is None:
                best, best_ep = vm["r_disp"], ep
                best_state = {k: v.detach().clone() for k, v in evm().state_dict().items()}
            if ep % (a.eval_every * 20) == 0:
                print(f"  ep {ep} loss {tot / n:.4f} val r_disp {vm['r_disp']:.3f} skill {vm['skill']:.3f} "
                      f"mae {vm['mae']:.4f} (zero {vm['zero_mae']:.4f}) best {best:.3f}@{best_ep} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    model.load_state_dict(best_state)  # (the EMA weights when --ema is set)
    val_items = predict(model, vl, device, scale, cell_scale, planar)
    test_items = predict(model, sl, device, scale, cell_scale, planar)
    vm, tm = metrics(val_items, r_eval), metrics(test_items, r_eval)
    if r_eval is not None:  # also on the model's own loss region
        vm.update({f"own_{k}": v for k, v in metrics(val_items).items()})
        tm.update({f"own_{k}": v for k, v in metrics(test_items).items()})
    out = dict(run_key=run_key, config=cfg, best_epoch=best_ep, val=vm, test=tm, curve=curve,
               train_time_s=time.time() - t0)
    json.dump(out, open(res_path, "w"), indent=1)
    if a.save_preds:
        os.makedirs(os.path.join(a.out_dir, "preds"), exist_ok=True)
        torch.save(dict(val=val_items, test=test_items, scale=scale, state=best_state, config=cfg),
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
