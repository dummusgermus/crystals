"""Delivery training config and global-loss v2 training for predictions_new.

Production settings (2026-08):
  * Point graphs: cutoff_k=13, edge_k=3 (from graph-size + loss sweeps)
  * Point CGCNN: full-cell global loss (target_mode=full)
  * Point Transformer: graph global loss, λ=0.02
  * Planar CGCNN: graph global loss, λ=0.005
  * Planar Transformer: graph global loss, λ=0.01
  * Within-group 90/10 split (seed 42), checkpoint by val R_tot on train target
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_gated_model_from_dataset, build_graph_transformer_from_dataset
from train_cgcnn_global_v2 import default_total_loss_config
from train_single import (
    TotalLossConfig,
    combine_atom_total_loss,
    ensure_graph_delta_field,
    evaluate_with_total_energy,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    per_graph_total_loss_weights,
    per_graph_total_scaled_loss,
    select_total_targets,
    set_seed,
    summarize_split,
    within_group_train_val_indices,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

DELIVERY_SEED = 42
DELIVERY_VAL_FRACTION = 0.1
DELIVERY_VERSION = "global_v2_k13"

# Point defect graph shell (k=13, edge_k=3 from sweeps).
POINT_CUTOFF_K = 13
POINT_EDGE_K = 3

# Best λ from grouped 70/15/15 sweeps (CGCNN follow-up 300 ep; Transformer 1000 ep).
LAMBDA_CGCNN_BY_DOMAIN = {
    "point": 0.01,
    "planar": 0.005,
}
LAMBDA_TRANSFORMER_BY_DOMAIN = {
    "point": 0.02,
    "planar": 0.01,
}
LAMBDA_BY_MODEL = {
    "cgcnn": LAMBDA_CGCNN_BY_DOMAIN,
    "transformer": LAMBDA_TRANSFORMER_BY_DOMAIN,
}

# Global-loss target: point CGCNN uses full-cell ΔE; all others use graph sum.
TOTAL_TARGET_MODE_BY_MODEL = {
    ("cgcnn", "point"): "full",
    ("cgcnn", "planar"): "graph",
    ("transformer", "point"): "graph",
    ("transformer", "planar"): "graph",
}


def lambda_tot_for(model_kind: str, domain: str) -> float:
    if model_kind not in LAMBDA_BY_MODEL:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    if domain not in ("point", "planar"):
        raise ValueError(f"Unknown domain: {domain}")
    return LAMBDA_BY_MODEL[model_kind][domain]


def total_loss_config_for(model_kind: str, domain: str) -> TotalLossConfig:
    cfg = default_total_loss_config(lambda_tot_for(model_kind, domain))
    cfg.target_mode = TOTAL_TARGET_MODE_BY_MODEL[(model_kind, domain)]
    return cfg


TOTALS_DATASETS = {
    "point": os.path.join(
        ROOT, "adv_datasets", "cycle34_residual_totals_k13_dataset.pt"
    ),
    "planar": os.path.join(
        ROOT, "planar_pyg_dataset_residual_c14c15_totals.pt"
    ),
}

EXPORT_DATASETS = {
    "point": os.path.join(ROOT, "adv_datasets", "cycle34_residual_k13_dataset.pt"),
    "planar": os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15.pt"),
}

SPLIT_JSON_DEFAULT = os.path.join(ROOT, "predictions_new", "delivery_split_indices.json")

CGCNN_CONFIG = dict(
    hidden_dim=128,
    num_layers=2,
    dropout=0.0,
    use_batch_norm=False,
    activation="silu",
    bidirectional=True,
    lr=2e-3,
    weight_decay=0.0,
    batch_size=8,
)

TRANSFORMER_CONFIG = dict(
    hidden_dim=128,
    num_layers=4,
    num_heads=4,
    dropout=0.1,
    attention_dropout=0.1,
    activation="gelu",
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=8,
    warmup_iters=20,
    min_lr=0.0,
    gradient_norm=1.0,
    scheduler="cosine_warmup",
)

CHECKPOINTS = {
    "point_cgcnn": os.path.join(ROOT, "cgcnn_defect_residual_global_v2_model.pt"),
    "planar_cgcnn": os.path.join(
        ROOT, "cgcnn_planar_residual_global_v2_model.pt"
    ),
    "point_transformer": os.path.join(
        ROOT, "transformer_graph_defect_residual_global_v2_model.pt"
    ),
    "planar_transformer": os.path.join(
        ROOT, "transformer_graph_planar_residual_global_v2_model.pt"
    ),
}

ALL_MODEL_KEYS = tuple(CHECKPOINTS.keys())
CURVES_JSON = os.path.join(ROOT, "delivery_global_v2_curves.json")
BENCHMARK_JSON = os.path.join(ROOT, "delivery_inference_benchmark.json")


class CosineWithWarmupLR:
    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        lr: float,
        lr_decay_iters: int,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr = lr
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

    def __call__(self, epoch: int) -> None:
        lr = self._get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self, epoch: int) -> float:
        step = epoch + 1
        if step <= self.warmup_iters:
            return self.lr * step / max(self.warmup_iters, 1)
        if step > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (step - self.warmup_iters) / max(
            self.lr_decay_iters - self.warmup_iters, 1
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)


def build_split_record(dataset, domain: str, dataset_path: str) -> dict:
    train_idx, val_idx = within_group_train_val_indices(
        dataset, DELIVERY_SEED, val_fraction=DELIVERY_VAL_FRACTION
    )
    return {
        "domain": domain,
        "dataset_path": dataset_path,
        "export_dataset_path": EXPORT_DATASETS[domain],
        "n_graphs": len(dataset),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_indices": [int(i) for i in train_idx.tolist()],
        "val_indices": [int(i) for i in val_idx.tolist()],
    }


def save_delivery_split(split_payload: dict, path: str = SPLIT_JSON_DEFAULT) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(split_payload, fh, indent=2)
    return path


def load_delivery_split(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def val_index_set(split_payload: dict, domain: str) -> set[int]:
    block = split_payload[domain]
    return set(int(i) for i in block["val_indices"])


def build_or_update_split_json(
    *,
    path: str = SPLIT_JSON_DEFAULT,
    force: bool = False,
) -> dict:
    if os.path.isfile(path) and not force:
        return load_delivery_split(path)

    payload = {
        "seed": DELIVERY_SEED,
        "val_fraction": DELIVERY_VAL_FRACTION,
        "splitter": "within_group_train_val_indices",
        "version": DELIVERY_VERSION,
        "loss": "global_v2",
        "checkpoint_metric": "val_r_tot_median",
        "point_cutoff_k": POINT_CUTOFF_K,
        "point_edge_k": POINT_EDGE_K,
        "lambda_by_model": LAMBDA_BY_MODEL,
        "total_target_mode_by_model": {
            f"{kind}_{domain}": TOTAL_TARGET_MODE_BY_MODEL[(kind, domain)]
            for domain in ("point", "planar")
            for kind in ("cgcnn", "transformer")
        },
        "totals_datasets": TOTALS_DATASETS,
        "export_datasets": EXPORT_DATASETS,
    }
    for domain in ("point", "planar"):
        ds_path = TOTALS_DATASETS[domain]
        if not os.path.isfile(ds_path):
            raise FileNotFoundError(f"Missing totals dataset: {ds_path}")
        dataset = torch.load(ds_path, weights_only=False)
        payload[domain] = build_split_record(dataset, domain, ds_path)
    save_delivery_split(payload, path)
    return payload


def _build_model(model_kind: str, dataset, config: Dict):
    if model_kind == "cgcnn":
        return build_gated_model_from_dataset(
            dataset,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            use_batch_norm=config["use_batch_norm"],
            activation=config["activation"],
            bidirectional=config["bidirectional"],
        )
    if model_kind == "transformer":
        return build_graph_transformer_from_dataset(
            dataset,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            attention_dropout=config["attention_dropout"],
            activation=config["activation"],
        )
    raise ValueError(f"Unknown model_kind: {model_kind}")


def train_delivery_global_v2(
    *,
    domain: str,
    model_kind: str,
    device: torch.device,
    epochs: int,
    metric: str,
    checkpoint_path: str,
    split_payload: dict,
    curves_path: Optional[str] = None,
) -> Dict:
    if domain not in ("point", "planar"):
        raise ValueError(domain)
    if model_kind not in ("cgcnn", "transformer"):
        raise ValueError(model_kind)

    name = f"{domain}_{model_kind}"
    dataset_path = TOTALS_DATASETS[domain]
    dataset = torch.load(dataset_path, weights_only=False)
    ensure_graph_delta_field(dataset)

    train_idx = np.array(split_payload[domain]["train_indices"], dtype=int)
    val_idx = np.array(split_payload[domain]["val_indices"], dtype=int)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]

    print(
        f"\n=== {name} | delivery {DELIVERY_VERSION} | train={len(train_set)} "
        f"val={len(val_set)} lambda={total_cfg.lambda_tot:g} "
        f"target_mode={total_cfg.target_mode} ===",
        flush=True,
    )
    summarize_split(f"[{name}] Train", train_set)
    summarize_split(f"[{name}] Val", val_set)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

    train_targets = torch.cat([d.y for d in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    model = _build_model(model_kind, dataset, config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] params={n_params:,}", flush=True)

    if model_kind == "cgcnn":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
        )
        scheduler_mode = "plateau"
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        scheduler = CosineWithWarmupLR(
            optimizer=optimizer,
            warmup_iters=config["warmup_iters"],
            lr=config["lr"],
            lr_decay_iters=epochs,
            min_lr=config["min_lr"],
        )
        scheduler_mode = "cosine"

    curves = {
        "train_mae": [],
        "val_mae": [],
        "val_r_tot_median": [],
        "val_abs_total_err_eV": [],
    }
    best_score = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        if scheduler_mode == "cosine":
            scheduler(epoch - 1)

        model.train()
        t0 = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            y_norm = (batch.y - target_mean) / target_std
            if metric == "mae":
                atom_loss = per_graph_mae_loss(pred, y_norm, batch.batch)
            else:
                atom_loss = per_graph_mse_loss(pred, y_norm, batch.batch)

            pred_denorm = pred * target_std + target_mean
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
            loss.backward()
            if model_kind == "transformer" and config.get("gradient_norm"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float(config["gradient_norm"])
                )
            optimizer.step()

        train_atom, train_tot = evaluate_with_total_energy(
            model,
            train_loader,
            device,
            target_mean,
            target_std,
            total_target_mode=total_cfg.target_mode,
        )
        val_atom, val_tot = evaluate_with_total_energy(
            model,
            val_loader,
            device,
            target_mean,
            target_std,
            total_target_mode=total_cfg.target_mode,
        )

        val_mae = metric_value(val_atom, metric)
        curves["train_mae"].append(metric_value(train_atom, metric))
        curves["val_mae"].append(val_mae)
        curves["val_r_tot_median"].append(val_tot.median_rel_total_err_pct)
        curves["val_abs_total_err_eV"].append(val_tot.mean_abs_total_err_eV)

        if scheduler_mode == "plateau":
            scheduler.step(val_mae)

        score = val_tot.median_rel_total_err_pct
        if score < best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            dt = time.time() - t0
            print(
                f"[{name}] ep {epoch:04d} | val MAE {val_mae:.4f} | "
                f"val R_tot {val_tot.median_rel_total_err_pct:.1f}% | "
                f"val |dE| {val_tot.mean_abs_total_err_eV:.4f} eV | {dt:.1f}s",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_atom, final_tot = evaluate_with_total_energy(
        model,
        val_loader,
        device,
        target_mean,
        target_std,
        total_target_mode=total_cfg.target_mode,
    )
    _, final_tot_extra = evaluate_with_total_energy(
        model,
        val_loader,
        device,
        target_mean,
        target_std,
        total_target_mode=extra_eval_mode,
    )

    ckpt = {
        "model_state": best_state,
        "architecture": "graph" if model_kind == "transformer" else "cgcnn",
        "config": config,
        "target_mean": float(target_mean.cpu()),
        "target_std": float(target_std.cpu()),
        "domain": domain,
        "model_kind": model_kind,
        "loss_mode": DELIVERY_VERSION,
        "lambda_tot": total_cfg.lambda_tot,
        "checkpoint_metric": "val_r_tot_median",
        "best_val_r_tot_median": best_score,
        "best_val_mae": metric_value(final_atom, metric),
        "epochs": epochs,
        "seed": DELIVERY_SEED,
        "val_fraction": DELIVERY_VAL_FRACTION,
        "dataset": dataset_path,
        "point_cutoff_k": POINT_CUTOFF_K if domain == "point" else None,
        "point_edge_k": POINT_EDGE_K if domain == "point" else None,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "train_indices": split_payload[domain]["train_indices"],
        "val_indices": split_payload[domain]["val_indices"],
        "split_json": SPLIT_JSON_DEFAULT,
        "total_loss_config": total_cfg.__dict__,
        "num_parameters": n_params,
    }
    torch.save(ckpt, checkpoint_path)
    print(f"[{name}] Saved checkpoint -> {checkpoint_path}", flush=True)

    result = {
        "domain": domain,
        "model_kind": model_kind,
        "checkpoint": checkpoint_path,
        "lambda_tot": total_cfg.lambda_tot,
        "total_target_mode": total_cfg.target_mode,
        "best_val_r_tot_median": best_score,
        "best_val_mae": ckpt["best_val_mae"],
        "final_val_r_tot_median": final_tot.median_rel_total_err_pct,
        f"final_val_r_tot_{extra_eval_mode}_median": (
            final_tot_extra.median_rel_total_err_pct
        ),
        "final_val_mae": metric_value(final_atom, metric),
        "n_train": len(train_set),
        "n_val": len(val_set),
        **curves,
    }
    if curves_path:
        os.makedirs(os.path.dirname(curves_path) or ".", exist_ok=True)
        with open(curves_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return result


def benchmark_delivery_inference(
    *,
    model_key: str,
    checkpoint_path: str,
    split_payload: dict,
    device: torch.device,
    warmup_batches: int = 3,
) -> dict:
    """Time forward passes on the delivery validation split."""
    domain, model_kind = model_key.split("_", 1)
    dataset_path = TOTALS_DATASETS[domain]
    dataset = torch.load(dataset_path, weights_only=False)
    val_idx = np.array(split_payload[domain]["val_indices"], dtype=int)
    val_set = [dataset[i] for i in val_idx]

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(ckpt["config"])
    model = _build_model(model_kind, dataset, config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    bs = int(config["batch_size"])
    loader = DataLoader(val_set, batch_size=bs, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            model(batch)
            if i + 1 >= warmup_batches:
                break

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_graphs = 0
    n_nodes = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            model(batch)
            n_graphs += int(batch.num_graphs)
            n_nodes += int(batch.num_nodes)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "model_key": model_key,
        "checkpoint": checkpoint_path,
        "dataset": dataset_path,
        "val_graphs": n_graphs,
        "val_nodes": n_nodes,
        "inference_batch_size": bs,
        "inference_wall_s": elapsed,
        "inference_ms_per_graph": 1000.0 * elapsed / max(n_graphs, 1),
        "inference_ms_per_node": 1000.0 * elapsed / max(n_nodes, 1),
        "inference_graphs_per_s": n_graphs / max(elapsed, 1e-9),
    }
