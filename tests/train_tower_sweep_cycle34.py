"""Sweep the number of towers (k=1,2,4,8) on cycle34_dataset.pt.

k=1 is the bidirectional base model (no towers, no mixing MLP).
k>1 uses the fused multi-tower message passing implementation.
All models use the same best hyperparameters, data split, and seed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_model_from_dataset, build_tower_model_from_dataset
from train_single import (
    evaluate,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    set_seed,
    summarize_split,
)

DEFAULT_DATASET = os.path.join("adv_datasets", "cycle34_dataset.pt")
DEFAULT_OUTPUT = "tower_sweep_cycle34.json"


def _train_one(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    metric: str,
) -> Dict[str, Any]:
    """Train *model* and return per-epoch curves + timing information."""

    model = model.to(device)
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
        description="Sweep number of towers (k=1 is base) on cycle34_dataset.pt."
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--tower-counts",
        type=str,
        default="1,2,4,8",
        help="Comma-separated tower counts to sweep. k=1 = base model (no towers).",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-batch-norm", action="store_true")
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tower_counts = [int(k.strip()) for k in args.tower_counts.split(",")]

    if not os.path.isfile(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Tower counts to sweep: {tower_counts}")

    dataset = torch.load(args.dataset, weights_only=False)

    set_seed(args.seed)
    train_idx, val_idx, test_idx = grouped_split_indices(dataset, args.seed)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    summarize_split("Train", train_set)
    summarize_split("Val", val_set)
    summarize_split("Test", test_set)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    train_targets = torch.cat([d.y for d in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    common_train_kwargs = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        metric=args.metric,
    )

    model_kwargs = dict(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        activation=args.activation,
    )

    all_curves: Dict[str, Dict[str, Any]] = {}

    for k in tower_counts:
        print("\n" + "=" * 60)
        if k == 1:
            label = "k=1 (base)"
            print(f"  Model: bidirectional base (no towers)")
        else:
            label = f"k={k}"
            print(f"  Model: bidirectional + {k}-tower MPNN  (tower_dim={args.hidden_dim // k})")
        print("=" * 60)

        set_seed(args.seed)
        if k == 1:
            model = build_model_from_dataset(dataset, bidirectional=True, **model_kwargs)
        else:
            model = build_tower_model_from_dataset(
                dataset, num_towers=k, bidirectional=True, **model_kwargs,
            )

        curves = _train_one(name=label, model=model, **common_train_kwargs)
        all_curves[label] = curves

    # summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for label, c in all_curves.items():
        final = c["test"][-1] if c["test"] else float("nan")
        print(
            f"  {label:20s} -> test MAE: {final:.4f}  "
            f"({c['total_train_time']:.0f}s, {c['num_parameters']:,} params)"
        )

    output = {
        "dataset_path": args.dataset,
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
        "tower_counts": tower_counts,
        "models": all_curves,
        "final_test_mae": {
            label: c["test"][-1] if c["test"] else float("nan")
            for label, c in all_curves.items()
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
