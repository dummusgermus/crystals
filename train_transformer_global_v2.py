"""Train Graph Transformer with global-loss v2 on totals datasets.

Grouped 70/15/15 split; global runs checkpoint by val R_tot median.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional

import torch
from torch_geometric.loader import DataLoader

from delivery_global_v2 import CosineWithWarmupLR
from gnn_models import build_graph_transformer_from_dataset
from train_cgcnn_global_v2 import default_total_loss_config
from train_single import (
    TotalLossConfig,
    combine_atom_total_loss,
    ensure_graph_delta_field,
    evaluate_with_total_energy,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    per_graph_total_loss_weights,
    per_graph_total_scaled_loss,
    select_total_targets,
    set_seed,
    summarize_split,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

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


def _train_one(
    *,
    domain: str,
    dataset,
    device: torch.device,
    epochs: int,
    metric: str,
    seed: int,
    config: Dict,
    use_total_loss: bool,
    lambda_tot: float,
    run_key: Optional[str] = None,
    checkpoint_metric: str = "mae",
    total_loss_config: Optional[TotalLossConfig] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict:
    loss_tag = "global_v2" if use_total_loss else "atom_only"
    name = run_key or (
        f"{domain}_transformer_global_v2_lambda_{lambda_tot:g}"
        if use_total_loss
        else f"{domain}_transformer_atom_only"
    )
    set_seed(seed)
    ensure_graph_delta_field(dataset)

    train_idx, val_idx, test_idx = grouped_split_indices(dataset, seed)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    print(
        f"\n=== {name} | train={len(train_set)} val={len(val_set)} test={len(test_set)} ===",
        flush=True,
    )
    summarize_split(f"[{name}] Train", train_set)
    summarize_split(f"[{name}] Val", val_set)
    summarize_split(f"[{name}] Test", test_set)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    train_targets = torch.cat([d.y for d in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    model = build_graph_transformer_from_dataset(
        dataset,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
        activation=config["activation"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] params={n_params:,}", flush=True)

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

    curves = {
        "train_mae": [],
        "val_mae": [],
        "test_mae": [],
        "train_r_tot_median": [],
        "val_r_tot_median": [],
        "test_r_tot_median": [],
        "train_r_tot_mean": [],
        "val_r_tot_mean": [],
        "test_r_tot_mean": [],
        "train_abs_total_err_eV": [],
        "val_abs_total_err_eV": [],
        "test_abs_total_err_eV": [],
    }
    best_val = float("inf")
    best_state = None
    ckpt_metric = checkpoint_metric.lower()
    if ckpt_metric not in {"mae", "r_tot"}:
        raise ValueError("checkpoint_metric must be 'mae' or 'r_tot'")

    eval_target_mode = (
        total_loss_config.target_mode
        if use_total_loss and total_loss_config is not None
        else "graph"
    )

    for epoch in range(1, epochs + 1):
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

            if use_total_loss:
                assert total_loss_config is not None
                pred_denorm = pred * target_std + target_mean
                targets = select_total_targets(
                    batch, target_mode=total_loss_config.target_mode
                )
                weights = per_graph_total_loss_weights(
                    batch,
                    targets,
                    max_delta_eV=total_loss_config.outlier_max_delta_eV,
                    max_mismatch_eV=total_loss_config.outlier_max_mismatch_eV,
                )
                tot_loss = per_graph_total_scaled_loss(
                    pred_denorm,
                    targets,
                    batch.batch,
                    scale_eps=total_loss_config.scale_eps,
                    loss_type=total_loss_config.loss_type,
                    huber_delta=total_loss_config.huber_delta,
                    weights=weights,
                )
                loss = combine_atom_total_loss(
                    atom_loss,
                    tot_loss,
                    lambda_tot=total_loss_config.lambda_tot,
                    balance_losses=total_loss_config.balance_losses,
                )
            else:
                loss = atom_loss

            loss.backward()
            if config.get("gradient_norm"):
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
            total_target_mode=eval_target_mode,
        )
        val_atom, val_tot = evaluate_with_total_energy(
            model,
            val_loader,
            device,
            target_mean,
            target_std,
            total_target_mode=eval_target_mode,
        )
        test_atom, test_tot = evaluate_with_total_energy(
            model,
            test_loader,
            device,
            target_mean,
            target_std,
            total_target_mode=eval_target_mode,
        )

        curves["train_mae"].append(metric_value(train_atom, metric))
        curves["val_mae"].append(metric_value(val_atom, metric))
        curves["test_mae"].append(metric_value(test_atom, metric))
        curves["train_r_tot_median"].append(train_tot.median_rel_total_err_pct)
        curves["val_r_tot_median"].append(val_tot.median_rel_total_err_pct)
        curves["test_r_tot_median"].append(test_tot.median_rel_total_err_pct)
        curves["train_r_tot_mean"].append(train_tot.mean_rel_total_err_pct)
        curves["val_r_tot_mean"].append(val_tot.mean_rel_total_err_pct)
        curves["test_r_tot_mean"].append(test_tot.mean_rel_total_err_pct)
        curves["train_abs_total_err_eV"].append(train_tot.mean_abs_total_err_eV)
        curves["val_abs_total_err_eV"].append(val_tot.mean_abs_total_err_eV)
        curves["test_abs_total_err_eV"].append(test_tot.mean_abs_total_err_eV)

        score = (
            curves["val_r_tot_median"][-1]
            if ckpt_metric == "r_tot"
            else curves["val_mae"][-1]
        )
        if score < best_val:
            best_val = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            dt = time.time() - t0
            print(
                f"[{name}] ep {epoch:04d} | val MAE {curves['val_mae'][-1]:.4f} | "
                f"val R_tot {curves['val_r_tot_median'][-1]:.1f}% | "
                f"test R_tot {curves['test_r_tot_median'][-1]:.1f}% | {dt:.1f}s",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    _, final_val_tot = evaluate_with_total_energy(
        model, val_loader, device, target_mean, target_std,
        total_target_mode=eval_target_mode,
    )
    _, final_test_tot = evaluate_with_total_energy(
        model, test_loader, device, target_mean, target_std,
        total_target_mode=eval_target_mode,
    )

    result = {
        **curves,
        "run_key": name,
        "domain": domain,
        "model": "transformer",
        "loss_mode": loss_tag,
        "use_total_loss": use_total_loss,
        "lambda_tot": lambda_tot if use_total_loss else 0.0,
        "checkpoint_metric": ckpt_metric,
        "best_val_score": best_val,
        "final_val_mae": curves["val_mae"][-1],
        "final_val_r_tot_median": final_val_tot.median_rel_total_err_pct,
        "final_test_mae": curves["test_mae"][-1],
        "final_test_r_tot_median": final_test_tot.median_rel_total_err_pct,
        "final_test_abs_total_err_eV": curves["test_abs_total_err_eV"][-1],
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "total_target_mode": eval_target_mode,
        "config": dict(config),
    }
    if total_loss_config is not None:
        result["total_loss_config"] = total_loss_config.__dict__

    if checkpoint_path:
        ckpt = {
            "model_state": best_state,
            "architecture": "graph",
            "config": config,
            "target_mean": float(target_mean.cpu()),
            "target_std": float(target_std.cpu()),
            "domain": domain,
            "model_kind": "transformer",
            "loss_mode": loss_tag,
            "lambda_tot": lambda_tot if use_total_loss else 0.0,
            "checkpoint_metric": ckpt_metric,
            "best_val_score": best_val,
            "final_test_r_tot_median": result["final_test_r_tot_median"],
            "final_test_mae": result["final_test_mae"],
            "epochs": epochs,
            "seed": seed,
            "split": "grouped 70/15/15",
            "run_key": name,
            "num_parameters": n_params,
        }
        if total_loss_config is not None:
            ckpt["total_loss_config"] = total_loss_config.__dict__
        torch.save(ckpt, checkpoint_path)
        result["checkpoint"] = checkpoint_path
        print(f"[{name}] Saved checkpoint -> {checkpoint_path}", flush=True)

    return result
