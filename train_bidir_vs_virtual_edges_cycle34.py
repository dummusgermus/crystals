"""Compare bidirectional GNN on cycle34 vs cycle34 + virtual edges.

Trains both variants for a fixed number of epochs, records per-epoch
train/val/test MAE and per-epoch wall-clock times, then writes a JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_model_from_dataset
from train_single import (
    evaluate,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    set_seed,
    summarize_split,
)

DEFAULT_BASE_DATASET = os.path.join("adv_datasets", "cycle34_dataset.pt")
DEFAULT_VE_DATASET = os.path.join("adv_datasets", "virtual_edge_cycle34_dataset.pt")
DEFAULT_OUTPUT = "bidir_vs_virtual_edges_cycle34.json"


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
) -> Dict[str, Any]:
    """Train a bidirectional model on *dataset* and return curves + timing."""

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
        bidirectional=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    train_curve: List[float] = []
    val_curve: List[float] = []
    test_curve: List[float] = []
    epoch_times: List[float] = []

    total_t0 = time.time()

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

        dt = time.time() - t0
        epoch_times.append(dt)

        scheduler.step(val_curve[-1])

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{name}] Epoch {epoch:03d} | "
            f"train {metric.upper()} {train_curve[-1]:.4f} | "
            f"val {metric.upper()} {val_curve[-1]:.4f} | "
            f"test {metric.upper()} {test_curve[-1]:.4f} | "
            f"lr {current_lr:.1e} | "
            f"time {dt:.1f}s"
        )

    total_time = time.time() - total_t0

    return {
        "train": train_curve,
        "val": val_curve,
        "test": test_curve,
        "epoch_times": epoch_times,
        "total_train_time": total_time,
        "num_parameters": n_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare bidirectional GNN on cycle34 vs cycle34 + virtual edges."
        )
    )
    parser.add_argument("--base-dataset", type=str, default=DEFAULT_BASE_DATASET)
    parser.add_argument("--ve-dataset", type=str, default=DEFAULT_VE_DATASET)
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

    for path in (args.base_dataset, args.ve_dataset):
        if not os.path.isfile(path):
            raise SystemExit(f"Dataset not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")

    common = dict(
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

    # --- Model 1: bidirectional on cycle34 ---
    print("\n" + "=" * 60)
    print(f"  Model: bidirectional (cycle34)")
    print(f"  Dataset: {args.base_dataset}")
    print("=" * 60)
    base_ds = torch.load(args.base_dataset, weights_only=False)
    curves_base = _train_one(name="bidir", dataset=base_ds, **common)

    # --- Model 2: bidirectional on cycle34 + virtual edges ---
    print("\n" + "=" * 60)
    print(f"  Model: bidirectional + virtual edges (cycle34)")
    print(f"  Dataset: {args.ve_dataset}")
    print("=" * 60)
    ve_ds = torch.load(args.ve_dataset, weights_only=False)
    curves_ve = _train_one(name="bidir+ve", dataset=ve_ds, **common)

    # --- Summary ---
    def _final(c: Dict) -> float:
        return float(c["test"][-1]) if c["test"] else float("nan")

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Bidir base          -> test MAE: {_final(curves_base):.4f}  "
          f"({curves_base['total_train_time']:.0f}s, "
          f"{curves_base['num_parameters']:,} params)")
    print(f"  Bidir + virt. edges -> test MAE: {_final(curves_ve):.4f}  "
          f"({curves_ve['total_train_time']:.0f}s, "
          f"{curves_ve['num_parameters']:,} params)")

    output = {
        "base_dataset": args.base_dataset,
        "ve_dataset": args.ve_dataset,
        "epochs": args.epochs,
        "metric": args.metric,
        "seed": args.seed,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "use_batch_norm": args.use_batch_norm,
            "activation": args.activation,
        },
        "models": {
            "bidirectional": curves_base,
            "bidirectional_virtual_edges": curves_ve,
        },
        "final_test_mae": {
            "bidirectional": _final(curves_base),
            "bidirectional_virtual_edges": _final(curves_ve),
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
