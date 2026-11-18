"""Atom-type encoding sweep on point / planar residual datasets.

Uses the production global-v2 delivery setup (delivery_global_v2.py: model
configs, per-model lambda, total-target mode, loss balancing, val-R_tot
checkpointing) but with a held-out test split so encodings can be compared.

Splits
------
random : 70/15/15 random graph split (same as the lambda sweeps, whose grouped
         split degenerates to random on these datasets). Seeded, so every
         encoding sees exactly the same split for a given seed.
chem   : 70/15/15 split over A-B element pairs (all graphs of a chemistry go
         to one side) -> tests generalisation to unseen compounds.

Example::

    python train_atom_encoding_sweep.py --domain point --model cgcnn \
        --encoding elem_props --seed 0 --epochs 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from atom_encodings import ENCODINGS, element_pair_key, encode_dataset
from edge_features import encode_edges
from positional_encodings import (
    apply_pe,
    build_pair_bias_transformer,
    has_pair_bias,
    lap_sign_flip,
    load_or_build_cache,
    parse_pe,
)
from delivery_global_v2 import (
    CGCNN_CONFIG,
    TARGET_NORM,
    TOTALS_DATASETS,
    TRANSFORMER_CONFIG,
    CosineWithWarmupLR,
    _build_model,
    total_loss_config_for,
)
from train_single import (
    combine_atom_total_loss,
    ensure_graph_delta_field,
    evaluate_with_total_energy,
    per_graph_mae_loss,
    per_graph_total_loss_weights,
    per_graph_total_scaled_loss,
    select_total_targets,
    set_seed,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR_DEFAULT = os.path.join(ROOT, "atom_encoding_sweep")

# --- Weights & Biases -------------------------------------------------------
# wandb lives outside the venv; appended (not prepended) so it never shadows
# the venv's torch/numpy deps. $HOME is read-only on the cluster -> redirect
# all wandb dirs. API key: WANDB_API_KEY (from ~/.my_profile, exported by the
# slurm script). Every wandb call is best-effort: failures never stop training.
WANDB_PROJECT_DEFAULT = "yara-crystals"
WANDB_PKG_DIR = "/work/log1/erik.mueller/python-packages"
WANDB_HOME = "/work/log1/erik.mueller/.wandb"


class WandbLogger:
    def __init__(self, enabled: bool, project: str, name: str, config: dict,
                 group: str, job_type: str, tags: List[str]) -> None:
        self.run = None
        if not enabled:
            return
        try:
            if WANDB_PKG_DIR not in sys.path and os.path.isdir(WANDB_PKG_DIR):
                sys.path.append(WANDB_PKG_DIR)
            for var, sub in (("WANDB_DIR", ""), ("WANDB_CACHE_DIR", "cache"),
                             ("WANDB_CONFIG_DIR", "config"), ("WANDB_DATA_DIR", "data")):
                os.environ.setdefault(var, os.path.join(WANDB_HOME, sub))
                os.makedirs(os.environ[var], exist_ok=True)
            import wandb

            self.run = wandb.init(
                project=project, name=name, config=config, group=group,
                job_type=job_type, tags=tags, reinit="finish_previous",
                settings=wandb.Settings(init_timeout=120, silent=True),
            )
            print(f"[wandb] logging to {self.run.url}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] disabled ({type(exc).__name__}: {exc})", flush=True)
            self.run = None

    def log(self, data: dict, step: Optional[int] = None) -> None:
        if self.run is not None:
            try:
                self.run.log(data, step=step)
            except Exception as exc:  # noqa: BLE001
                print(f"[wandb] log failed: {exc}", flush=True)

    def summary(self, data: dict) -> None:
        if self.run is not None:
            try:
                self.run.summary.update(data)
            except Exception as exc:  # noqa: BLE001
                print(f"[wandb] summary failed: {exc}", flush=True)

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:  # noqa: BLE001
                pass


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _flatten(prefix: str, d: dict) -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(key, v))
        else:
            out[key] = v
    return out


def random_split(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1000 + seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    return idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]


def chem_split(dataset, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: Dict[str, List[int]] = {}
    for i, d in enumerate(dataset):
        groups.setdefault(element_pair_key(d), []).append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(2000 + seed)
    rng.shuffle(keys)
    n_tr, n_va = int(0.7 * len(keys)), int(0.15 * len(keys))
    pick = lambda ks: np.array(sorted(i for k in ks for i in groups[k]), dtype=int)
    return pick(keys[:n_tr]), pick(keys[n_tr : n_tr + n_va]), pick(keys[n_tr + n_va :])


def cv_split(dataset, fold: int, n_folds: int = 5, n_train_compounds=None):
    """Compound-grouped K-fold split (GRACE study): test = compounds of fold *fold*;
    15 % of the remaining compounds -> val; optional subsample of the train compounds
    (learning curve). Fold assignment is fixed (independent of the model seed)."""
    groups: Dict[str, List[int]] = {}
    for i, d in enumerate(dataset):
        groups.setdefault(element_pair_key(d), []).append(i)
    keys = sorted(groups)
    np.random.default_rng(5000).shuffle(keys)
    test_k = keys[fold::n_folds]
    rest = [k for k in keys if k not in set(test_k)]
    np.random.default_rng(6000 + fold).shuffle(rest)
    n_va = int(round(0.15 * len(rest)))
    val_k, train_k = rest[:n_va], rest[n_va:]
    if n_train_compounds is not None:
        train_k = list(np.random.default_rng(7000 + fold).permutation(train_k))[:n_train_compounds]
    pick = lambda ks: np.array(sorted(i for k in ks for i in groups[k]), dtype=int)
    return pick(train_k), pick(val_k), pick(test_k)


def _tot_dict(tot) -> dict:
    return {
        "r_tot_median": tot.median_rel_total_err_pct,
        "r_tot_mean": tot.mean_rel_total_err_pct,
        "abs_total_err_eV": tot.mean_abs_total_err_eV,
    }


def run(args) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain, model_kind, encoding = args.domain, args.model, args.encoding
    pe = args.pe
    run_key = (f"{args.split}_{domain}_{model_kind}_{encoding}"
               + (f"_pe-{pe}" if pe is not None else "")
               + (f"_v-{args.variant}" if args.variant else "")
               + (f"_sp{args.split_seed}" if args.split_seed is not None else "")
               + f"_s{args.seed}" + (f"_{args.tag}" if args.tag else ""))
    out_json = os.path.join(args.out_dir, "results", f"{run_key}.json")
    if os.path.isfile(out_json) and not args.force:
        print(f"[skip] {out_json} exists", flush=True)
        with open(out_json, encoding="utf-8") as fh:
            return json.load(fh)

    raw = torch.load(args.dataset_path or TOTALS_DATASETS[domain], weights_only=False)
    ensure_graph_delta_field(raw)
    dataset = encode_dataset(raw, encoding)
    if args.edge is None:  # default: production setting per model
        args.edge = (CGCNN_CONFIG if model_kind == "cgcnn" else TRANSFORMER_CONFIG).get("edge", "raw")
    dataset = encode_edges(dataset, args.edge)
    k_lap = 0
    if pe is not None:
        if model_kind == "cgcnn" and has_pair_bias(pe):
            raise SystemExit(f"pair-bias PE {pe!r} is Transformer-only")
        dataset, _pad = apply_pe(dataset, load_or_build_cache(raw, domain), pe)
        k_lap = parse_pe(pe).get("lap", 0)
    split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.split == "cv5":
        tr, va, te = cv_split(dataset, args.fold, 5, args.n_train_compounds)
    elif args.split == "random":
        tr, va, te = random_split(len(dataset), split_seed)
    else:
        tr, va, te = chem_split(dataset, split_seed)
    train_set = [dataset[i] for i in tr]
    val_set = [dataset[i] for i in va]
    test_set = [dataset[i] for i in te]

    config = dict(CGCNN_CONFIG if model_kind == "cgcnn" else TRANSFORMER_CONFIG)
    # optional training-recipe overrides (plateau study); None = production default
    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch_norm:
        config["use_batch_norm"] = True
    if args.hidden is not None:
        config["hidden_dim"] = args.hidden
    if args.layers is not None:
        config["num_layers"] = args.layers
    if args.heads is not None:
        config["num_heads"] = args.heads
    # defaults follow production (delivery_global_v2): CGCNN now cosine + clip, see plateau study
    sched_kind = args.sched or ("plateau" if config.get("scheduler") == "plateau" else "cosine")
    if args.target_norm is None:
        args.target_norm = TARGET_NORM[(model_kind, domain)]
    recipe = {
        "sched": sched_kind,
        "plateau_patience": args.patience,
        "min_lr": args.min_lr if args.min_lr is not None else config.get("min_lr", 1e-6),
        "warmup_epochs": args.warmup if args.warmup is not None else config.get("warmup_iters", 0),
        "grad_clip": args.clip if args.clip is not None else float(config.get("gradient_norm") or 0.0),
        "atom_loss": args.atom_loss,
        "mse_warmup_epochs": args.mse_warmup,
        "target_norm": args.target_norm,
        "edge": args.edge,
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "num_heads": config.get("num_heads"),
        "weight_decay": config["weight_decay"],
    }
    total_cfg = total_loss_config_for(model_kind, domain)
    # loss-sweep overrides (item 2); None = production default
    if args.lambda_tot is not None:
        total_cfg.lambda_tot = args.lambda_tot
    if args.tot_target is not None:
        total_cfg.target_mode = args.tot_target
    if args.tot_loss is not None:
        total_cfg.loss_type = args.tot_loss
    if args.tot_scale_eps is not None:
        total_cfg.scale_eps = args.tot_scale_eps
    epochs = args.epochs
    print(
        f"=== {run_key} | train={len(train_set)} val={len(val_set)} "
        f"test={len(test_set)} | x_dim={dataset[0].x.size(-1)} "
        f"edge_dim={dataset[0].edge_attr.size(-1)} | lambda={total_cfg.lambda_tot} "
        f"target={total_cfg.target_mode} | epochs={epochs} | {device} ===",
        flush=True,
    )

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    eval_bs = 32
    val_loader = DataLoader(val_set, batch_size=eval_bs, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=eval_bs, shuffle=False)

    y_tr = torch.cat([d.y for d in train_set]).view(-1)
    if args.target_norm == "robust":
        # median / MAD scaling: heavy-tailed residuals (planar: |y| p50 3.8 meV, std 139 meV)
        # make std-scaled targets ~0.03 for most atoms -> MAE plateau at "predict median".
        t_mean = y_tr.median().to(device)
        t_std = (1.4826 * (y_tr - y_tr.median()).abs().median()).clamp(min=1e-6).to(device)
    else:
        t_mean = y_tr.mean().to(device)
        t_std = y_tr.std(unbiased=False).clamp(min=1e-6).to(device)

    if pe is not None and has_pair_bias(pe):
        model = build_pair_bias_transformer(dataset, config, pe).to(device)
    else:
        model = _build_model(model_kind, dataset, config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt_cls = torch.optim.Adam if model_kind == "cgcnn" else torch.optim.AdamW
    opt = opt_cls(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if sched_kind == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=recipe["plateau_patience"], min_lr=recipe["min_lr"]
        )
    elif sched_kind == "cosine":
        sched = CosineWithWarmupLR(
            optimizer=opt,
            warmup_iters=recipe["warmup_epochs"],
            lr=config["lr"],
            lr_decay_iters=epochs,
            min_lr=recipe["min_lr"],
        )
    else:  # constant (optional linear warmup)
        sched = CosineWithWarmupLR(optimizer=opt, warmup_iters=recipe["warmup_epochs"], lr=config["lr"],
                                   lr_decay_iters=10**9, min_lr=config["lr"])
        sched._get_lr = lambda e, _s=sched: _s.lr * min(1.0, (e + 1) / max(_s.warmup_iters, 1))

    wb = WandbLogger(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        name=run_key,
        group=f"{args.group}-{args.split}" + (f"-{args.tag}" if args.tag else ""),
        job_type=f"{domain}_{model_kind}",
        tags=[f"{args.group}-sweep", args.split, domain, model_kind, encoding]
        + ([f"pe:{pe}"] if pe is not None else []) + ([args.tag] if args.tag else []),
        config={
            # experiment
            "experiment": f"{args.group}_sweep".replace("-", "_"),
            "run_key": run_key,
            "positional_encoding": pe or "none",
            **{f"recipe/{k}": v for k, v in recipe.items()},
            "lap_sign_flip": bool(k_lap),
            "domain": domain,
            "model_kind": model_kind,
            "architecture": "GatedGNN (CGCNN-style)" if model_kind == "cgcnn" else "GraphTransformer",
            "encoding": encoding,
            "split": args.split,
            "split_fractions": "70/15/15",
            "seed": args.seed,
            "epochs": epochs,
            "checkpoint_metric": "val_r_tot_median",
            # model + optimiser hyperparameters (delivery_global_v2 config)
            **{f"model/{k}": v for k, v in config.items()},
            "optimizer": "Adam" if model_kind == "cgcnn" else "AdamW",
            "scheduler": sched_kind,
            "eval_batch_size": eval_bs,
            "atom_loss": "per_graph_mae (normalised targets)",
            # global-loss v2
            **_flatten("total_loss", total_cfg.__dict__),
            # data
            "dataset": args.dataset_path or TOTALS_DATASETS[domain],
            "x_dim": int(dataset[0].x.size(-1)),
            "edge_dim": int(dataset[0].edge_attr.size(-1)),
            "n_train": len(train_set),
            "n_val": len(val_set),
            "n_test": len(test_set),
            "target_mean": float(t_mean.cpu()),
            "target_std": float(t_std.cpu()),
            "n_params": n_params,
            "git_commit": _git_commit(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "device": str(device),
        },
    )

    ev = lambda loader, mode=total_cfg.target_mode: evaluate_with_total_energy(
        model, loader, device, t_mean, t_std, total_target_mode=mode, min_delta_eV=args.rtot_min_delta
    )

    curves = {"val_mae": [], "val_r_tot_median": []}
    best_rtot, best_rtot_state, best_rtot_ep = float("inf"), None, -1
    best_mae, best_mae_state, best_mae_ep = float("inf"), None, -1
    # Pareto front of (val MAE, val R_tot) epochs -> "gated" checkpoint chosen at the end:
    # best val R_tot among epochs with val MAE <= (1 + tol) * min val MAE.
    front = []  # list of (mae, rtot, epoch, state)
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        if sched_kind != "plateau":
            sched(epoch - 1)
        model.train()
        ep_loss, ep_gn, n_b = 0.0, 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            lap_sign_flip(batch, k_lap)
            opt.zero_grad(set_to_none=True)
            pred = model(batch)
            y_norm = (batch.y - t_mean) / t_std
            use_mse = epoch <= recipe["mse_warmup_epochs"]
            if recipe["atom_loss"] == "mae" and not use_mse:
                atom_loss = per_graph_mae_loss(pred, y_norm, batch.batch)
            else:
                err = pred.view(-1) - y_norm.view(-1)
                el = (err ** 2 if (use_mse or recipe["atom_loss"] == "mse")
                      else torch.nn.functional.huber_loss(pred.view(-1), y_norm.view(-1), reduction="none", delta=1.0))
                ng = int(batch.num_graphs)
                atom_loss = (torch.zeros(ng, device=el.device).index_add_(0, batch.batch, el)
                             / torch.bincount(batch.batch, minlength=ng).clamp(min=1)).mean()
            pred_denorm = pred * t_std + t_mean
            targets = select_total_targets(batch, target_mode=total_cfg.target_mode)
            weights = per_graph_total_loss_weights(
                batch,
                targets,
                max_delta_eV=total_cfg.outlier_max_delta_eV,
                max_mismatch_eV=total_cfg.outlier_max_mismatch_eV,
            )
            tot_loss = per_graph_total_scaled_loss(
                pred_denorm,
                targets,
                batch.batch,
                scale_eps=total_cfg.scale_eps,
                loss_type=total_cfg.loss_type,
                huber_delta=total_cfg.huber_delta,
                weights=weights,
            )
            loss = combine_atom_total_loss(
                atom_loss,
                tot_loss,
                lambda_tot=total_cfg.lambda_tot,
                balance_losses=total_cfg.balance_losses,
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(recipe["grad_clip"]) if recipe["grad_clip"] else float("inf")
            )
            opt.step()
            ep_loss += float(loss.detach())
            ep_gn += float(gn)
            n_b += 1

        val_atom, val_tot = ev(val_loader)
        if sched_kind == "plateau":
            sched.step(val_atom.mae)
        curves["val_mae"].append(val_atom.mae)
        curves.setdefault("train_loss", []).append(ep_loss / max(n_b, 1))
        curves.setdefault("grad_norm", []).append(ep_gn / max(n_b, 1))
        curves.setdefault("lr", []).append(opt.param_groups[0]["lr"])
        curves["val_r_tot_median"].append(val_tot.median_rel_total_err_pct)
        if math.isfinite(val_tot.median_rel_total_err_pct) and (
            val_tot.median_rel_total_err_pct < best_rtot
        ):
            best_rtot, best_rtot_ep = val_tot.median_rel_total_err_pct, epoch
            best_rtot_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        m_, r_ = val_atom.mae, val_tot.median_rel_total_err_pct
        if math.isfinite(m_) and math.isfinite(r_) and not any(fm <= m_ and fr <= r_ for fm, fr, _, _ in front):
            front = [f for f in front if not (m_ <= f[0] and r_ <= f[1])]
            front.append((m_, r_, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        if math.isfinite(val_atom.mae) and val_atom.mae < best_mae:
            best_mae, best_mae_ep = val_atom.mae, epoch
            best_mae_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        wb.log(
            {
                "val/mae": val_atom.mae,
                "val/r_tot_median": val_tot.median_rel_total_err_pct,
                "val/abs_total_err_eV": val_tot.mean_abs_total_err_eV,
                "val/best_r_tot_median": best_rtot,
                "val/best_mae": best_mae,
                "lr": opt.param_groups[0]["lr"],
                "train/loss": ep_loss / max(n_b, 1),
                "train/grad_norm": ep_gn / max(n_b, 1),
                "time/s_per_epoch": (time.time() - t_start) / epoch,
            },
            step=epoch,
        )
        if epoch % 25 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[{run_key}] ep {epoch:04d} | val MAE {val_atom.mae:.5f} | "
                f"val R_tot {val_tot.median_rel_total_err_pct:.1f}% | "
                f"best R_tot {best_rtot:.1f}%@{best_rtot_ep} | "
                f"{(time.time() - t_start) / epoch:.2f}s/ep",
                flush=True,
            )

    def eval_state(state) -> dict:
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        res = {}
        for name, loader in (("val", val_loader), ("test", test_loader)):
            atom, tot = ev(loader)
            alt_mode = "full" if total_cfg.target_mode == "graph" else "graph"
            _, tot_alt = ev(loader, alt_mode)
            res[name] = {
                "mae": atom.mae,
                "rmse": atom.rmse,
                **_tot_dict(tot),
                f"r_tot_median_{alt_mode}": tot_alt.median_rel_total_err_pct,
            }
        return res

    result = {
        "run_key": run_key,
        "split": args.split,
        "domain": domain,
        "model_kind": model_kind,
        "encoding": encoding,
        "pe": pe or "none",
        "recipe": recipe,
        "variant": args.variant or "default",
        "seed": args.seed,
        "epochs": epochs,
        "lambda_tot": total_cfg.lambda_tot,
        "total_loss": dict(total_cfg.__dict__),
        "total_target_mode": total_cfg.target_mode,
        "n_params": n_params,
        "x_dim": int(dataset[0].x.size(-1)),
        "edge_dim": int(dataset[0].edge_attr.size(-1)),
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "wall_s": time.time() - t_start,
        "ckpt_rtot": {"epoch": best_rtot_ep, **eval_state(best_rtot_state)},
        "ckpt_mae": {"epoch": best_mae_ep, **eval_state(best_mae_state)},
        "ckpt_gated": (lambda f: {"epoch": f[2], "mae_tol": args.ckpt_mae_tol, **eval_state(f[3])})(
            min((f for f in front if f[0] <= (1 + args.ckpt_mae_tol) * min(x[0] for x in front)),
                key=lambda f: f[1])) if front else None,
        "curves": curves,
    }
    result["split_seed"] = split_seed
    if args.save_preds:  # per-atom predictions for offline ensembling (ensemble study)
        def dump(state) -> dict:
            model.load_state_dict({k: v.to(device) for k, v in state.items()})
            model.eval()
            out = {}
            for name, loader, idx in (("val", val_loader, va), ("test", test_loader, te)):
                preds, ys, gids, tg, tf, off = [], [], [], [], [], 0
                with torch.no_grad():
                    for b in loader:
                        b = b.to(device)
                        preds.append((model(b) * t_std + t_mean).view(-1).double().cpu())
                        ys.append(b.y.view(-1).double().cpu())
                        gids.append((b.batch.view(-1) + off).cpu())
                        tg.append(select_total_targets(b, target_mode="graph").double().cpu())
                        tf.append(select_total_targets(b, target_mode="full").double().cpu())
                        off += b.num_graphs
                out[name] = {"pred": torch.cat(preds), "y": torch.cat(ys), "graph": torch.cat(gids),
                             "tot_graph": torch.cat(tg), "tot_full": torch.cat(tf),
                             "indices": torch.as_tensor(np.asarray(idx))}
            return out
        pred_path = os.path.join(args.out_dir, "preds", f"{run_key}.pt")
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        torch.save({"ckpt_rtot": dump(best_rtot_state), "ckpt_mae": dump(best_mae_state),
                    "own_target_mode": total_cfg.target_mode}, pred_path)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    if args.save_model:
        ckpt_dir = os.path.join(args.out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                "model_state": best_rtot_state,
                "config": config,
                "encoding": encoding,
                "pe": pe or "none",
                "target_mean": float(t_mean.cpu()),
                "target_std": float(t_std.cpu()),
                "total_loss_config": total_cfg.__dict__,
            },
            os.path.join(ckpt_dir, f"{run_key}.pt"),
        )
    wb.summary(
        {
            **_flatten("ckpt_rtot", result["ckpt_rtot"]),
            **_flatten("ckpt_mae", result["ckpt_mae"]),
            **(_flatten("ckpt_gated", result["ckpt_gated"]) if result["ckpt_gated"] else {}),
            "wall_s": result["wall_s"],
        }
    )
    wb.finish()
    r = result["ckpt_rtot"]["test"]
    print(
        f"[done] {run_key} | best val R_tot {best_rtot:.2f}%@{best_rtot_ep} | "
        f"test MAE {r['mae']:.5f} | test R_tot {r['r_tot_median']:.2f}% | "
        f"test |dE| {r['abs_total_err_eV']:.4f} eV | {result['wall_s']/60:.1f} min",
        flush=True,
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["point", "planar"], required=True)
    p.add_argument("--model", choices=["cgcnn", "transformer"], required=True)
    p.add_argument("--encoding", choices=ENCODINGS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["random", "chem", "cv5"], default="random")
    p.add_argument("--fold", type=int, default=0, help="cv5: test fold 0-4")
    p.add_argument("--n-train-compounds", type=int, default=None, help="cv5: subsample train compounds (learning curve)")
    p.add_argument("--dataset-path", default=None, help="override dataset (default: production totals dataset)")
    p.add_argument("--rtot-min-delta", type=float, default=1e-6,
                   help="R_tot only over graphs with |true dE| >= this (eV); 1e-6 = old behaviour")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--save-preds", action="store_true", help="dump per-atom val/test predictions (ensembling)")
    p.add_argument("--split-seed", type=int, default=None, help="data split seed (default: --seed)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--tag", default="", help="suffix for run name / W&B group (e.g. ep2000)")
    p.add_argument("--pe", default=None,
                   help="positional encoding spec (positional_encodings.py), e.g. none, rwse16, lap8+rwse16, spd+geo")
    p.add_argument("--group", default="atom-encoding", help="W&B group prefix / experiment name")
    # training-recipe overrides (defaults = production recipe)
    p.add_argument("--variant", default="", help="name of the recipe variant (run name / results key)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--sched", choices=["plateau", "cosine", "constant"], default=None)
    p.add_argument("--patience", type=int, default=8, help="ReduceLROnPlateau patience")
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None, help="linear LR warmup epochs (cosine/constant)")
    p.add_argument("--clip", type=float, default=None, help="grad-norm clip (0 = off)")
    p.add_argument("--atom-loss", choices=["mae", "huber", "mse"], default="mae")
    p.add_argument("--mse-warmup", type=int, default=0, help="use MSE atom loss for the first N epochs")
    p.add_argument("--batch-norm", action="store_true")
    p.add_argument("--target-norm", choices=["std", "robust"], default=None,
                   help="default: production TARGET_NORM per (model, domain)")
    p.add_argument("--lambda-tot", type=float, default=None, help="total-energy loss weight override")
    p.add_argument("--tot-target", choices=["graph", "full"], default=None, help="total-energy target mode override")
    p.add_argument("--tot-loss", choices=["huber", "mse"], default=None, help="total-energy loss type override")
    p.add_argument("--tot-scale-eps", type=float, default=None, help="relative-error floor (eV) override")
    p.add_argument("--edge", default=None, help="edge-distance representation: raw | rbf{K} (default: production)")
    p.add_argument("--hidden", type=int, default=None, help="hidden width override")
    p.add_argument("--layers", type=int, default=None, help="number of conv / attention layers override")
    p.add_argument("--heads", type=int, default=None, help="Transformer heads override")
    p.add_argument("--ckpt-mae-tol", type=float, default=0.2,
                   help="gated checkpoint: best val R_tot among epochs with val MAE <= (1+tol)*min")
    p.add_argument("--wandb-project", default=WANDB_PROJECT_DEFAULT)
    p.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    args = p.parse_args()
    if args.epochs is None:
        args.epochs = 300 if args.model == "cgcnn" else 1000
    run(args)


if __name__ == "__main__":
    main()
