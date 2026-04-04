"""Train the bidirectional GNN on the master-node-augmented cycle34 dataset.

The master node is excluded from loss computation and evaluation metrics
via the ``node_mask`` attribute stored in each graph by
``masternode_graph_maker.py``.

Outputs per-epoch train / val / test MAE to a JSON file for later plotting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_model_from_dataset
from train_single import Metrics, grouped_split_indices, set_seed, summarize_split

DEFAULT_DATASET = os.path.join("adv_datasets", "masternode_cycle34_dataset.pt")
DEFAULT_OUTPUT = "masternode_cycle34.json"


# ---------------------------------------------------------------------------
# Masked loss / metrics (ignore master-node predictions)
# ---------------------------------------------------------------------------

def _masked_per_graph_reduction(
    values: torch.Tensor,
    batch: torch.Tensor,
    mask: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    v = values[mask]
    b = batch[mask]
    sums = torch.zeros(num_graphs, device=v.device)
    counts = torch.zeros(num_graphs, device=v.device)
    sums.index_add_(0, b, v)
    counts.index_add_(0, b, torch.ones_like(v))
    return sums / counts.clamp(min=1.0)


def masked_per_graph_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred, target, batch, mask = (t.view(-1) for t in (pred, target, batch, mask))
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return torch.tensor(0.0, device=pred.device)
    abs_err = torch.abs(pred - target)
    return _masked_per_graph_reduction(abs_err, batch, mask, num_graphs).mean()


def masked_per_graph_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred, target, batch, mask = (t.view(-1) for t in (pred, target, batch, mask))
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return torch.tensor(0.0, device=pred.device)
    sq_err = (pred - target) ** 2
    return _masked_per_graph_reduction(sq_err, batch, mask, num_graphs).mean()


def masked_compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: torch.Tensor,
    mask: torch.Tensor,
) -> Metrics:
    pred, target, batch, mask = (t.view(-1) for t in (pred, target, batch, mask))
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return Metrics(mse=0.0, rmse=0.0, mae=0.0)
    sq = (pred - target) ** 2
    abs_err = torch.abs(pred - target)
    mse = _masked_per_graph_reduction(sq, batch, mask, num_graphs).mean().item()
    mae = _masked_per_graph_reduction(abs_err, batch, mask, num_graphs).mean().item()
    return Metrics(mse=mse, rmse=math.sqrt(mse), mae=mae)


def masked_evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> Metrics:
    model.eval()
    total_mse = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            pred_denorm = pred * target_std + target_mean
            m = masked_compute_metrics(pred_denorm, batch.y, batch.batch, batch.node_mask)
            total_mse += m.mse * batch.num_graphs
            total_rmse += m.rmse * batch.num_graphs
            total_mae += m.mae * batch.num_graphs
            total_graphs += batch.num_graphs
    if total_graphs == 0:
        return Metrics(mse=0.0, rmse=0.0, mae=0.0)
    return Metrics(
        mse=total_mse / total_graphs,
        rmse=total_rmse / total_graphs,
        mae=total_mae / total_graphs,
    )


def _metric_value(metrics: Metrics, metric: str) -> float:
    return {"mae": metrics.mae, "rmse": metrics.rmse, "mse": metrics.mse}[metric.lower()]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_one(
    name: str,
    dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    use_batch_norm: bool,
    activation: str,
    metric: str,
    seed: int,
) -> Dict[str, List[float]]:
    set_seed(seed)

    train_idx, val_idx, test_idx = grouped_split_indices(dataset, seed)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    summarize_split(f"[{name}] Train", train_set)
    summarize_split(f"[{name}] Val", val_set)
    summarize_split(f"[{name}] Test", test_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    real_targets = torch.cat([d.y[d.node_mask] for d in train_set], dim=0).view(-1)
    target_mean = real_targets.mean().to(device)
    target_std = real_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    model = build_model_from_dataset(
        dataset,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation,
        bidirectional=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6,
    )

    train_curve: List[float] = []
    val_curve: List[float] = []
    test_curve: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            y_norm = (batch.y - target_mean) / target_std
            mask = batch.node_mask
            if metric == "mae":
                loss = masked_per_graph_mae_loss(pred, y_norm, batch.batch, mask)
            else:
                loss = masked_per_graph_mse_loss(pred, y_norm, batch.batch, mask)
            loss.backward()
            optimizer.step()

        train_m = masked_evaluate(model, train_loader, device, target_mean, target_std)
        val_m = masked_evaluate(model, val_loader, device, target_mean, target_std)
        test_m = masked_evaluate(model, test_loader, device, target_mean, target_std)

        train_curve.append(_metric_value(train_m, metric))
        val_curve.append(_metric_value(val_m, metric))
        test_curve.append(_metric_value(test_m, metric))

        scheduler.step(val_curve[-1])

        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{name}] Epoch {epoch:03d} | "
            f"train {metric.upper()} {train_curve[-1]:.4f} | "
            f"val {metric.upper()} {val_curve[-1]:.4f} | "
            f"test {metric.upper()} {test_curve[-1]:.4f} | "
            f"lr {current_lr:.1e} | "
            f"time {dt:.1f}s"
        )

    return {"train": train_curve, "val": val_curve, "test": test_curve}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train bidirectional GNN on master-node-augmented cycle34 dataset.",
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-batch-norm", action="store_true")
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")

    dataset = torch.load(args.dataset, weights_only=False)

    t0 = time.time()
    print("\n" + "=" * 60)
    print("  Model: bidirectional + master node")
    print("=" * 60)

    curves = _train_one(
        name="bidir-masternode",
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        activation=args.activation,
        metric=args.metric,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed / 60:.1f} min")

    final_test = float(curves["test"][-1]) if curves["test"] else float("nan")

    output = {
        "dataset_path": args.dataset,
        "epochs": args.epochs,
        "metric": args.metric,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "use_batch_norm": args.use_batch_norm,
        "activation": args.activation,
        "model": "bidirectional_masternode",
        "curves": curves,
        "final_test_mae": final_test,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
