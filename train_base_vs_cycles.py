"""Train the base GNN on the plain dataset and the three cycle-feature variants.

Runs 300 epochs per dataset, records per-epoch test metrics, and writes
everything to a single JSON file for later plotting / analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_model_from_dataset
from train_single import (
    compute_metrics,
    evaluate,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    set_seed,
    summarize_split,
)

DATASETS: OrderedDict[str, str] = OrderedDict([
    ("base", "pyg_dataset.pt"),
    ("cycle3", os.path.join("adv_datasets", "cycle3_dataset.pt")),
    ("cycle34", os.path.join("adv_datasets", "cycle34_dataset.pt")),
    ("cycle345", os.path.join("adv_datasets", "cycle345_dataset.pt")),
])


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
    bidirectional: bool,
) -> Dict[str, List[float]]:
    """Train a single model and return per-epoch train / val / test curves."""

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

    train_targets = torch.cat([d.y for d in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    model = build_model_from_dataset(
        dataset,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation,
        bidirectional=bidirectional,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
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
            if metric == "mae":
                loss = per_graph_mae_loss(pred, y_norm, batch.batch)
            else:
                loss = per_graph_mse_loss(pred, y_norm, batch.batch)
            loss.backward()
            optimizer.step()

        train_m = evaluate(model, train_loader, device, target_mean, target_std)
        val_m = evaluate(model, val_loader, device, target_mean, target_std)
        test_m = evaluate(model, test_loader, device, target_mean, target_std)

        train_curve.append(metric_value(train_m, metric))
        val_curve.append(metric_value(val_m, metric))
        test_curve.append(metric_value(test_m, metric))

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base GNN on plain vs cycle-feature datasets."
    )
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
    parser.add_argument("--output", type=str, default="base_vs_cycles_curves.json")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Two directed edges per undirected edge (same .pt datasets).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results: Dict[str, Dict[str, List[float]]] = {}

    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping '{name}'.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Training on: {name}  ({path})")
        print(f"{'=' * 60}")

        dataset = torch.load(path, weights_only=False)
        curves = _train_one(
            name=name,
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
            bidirectional=args.bidirectional,
        )
        results[name] = curves

    output = {
        "metric": args.metric,
        "epochs": args.epochs,
        "seed": args.seed,
        "bidirectional": args.bidirectional,
        "curves": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved all curves to {args.output}")


if __name__ == "__main__":
    main()
