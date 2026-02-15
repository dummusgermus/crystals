from __future__ import annotations

import argparse
import json
import os
import time
from glob import glob
from typing import Dict, List

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


def train_base_model(
    dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, List[float] | float]:
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

    train_targets = torch.cat([data.y for data in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    model = build_model_from_dataset(
        dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        activation=args.activation,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    train_curve: List[float] = []
    val_curve: List[float] = []
    test_curve: List[float] = []

    best_val = float("inf")
    best_test = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            y_norm = (batch.y - target_mean) / target_std
            if args.metric == "mae":
                loss = per_graph_mae_loss(pred, y_norm, batch.batch)
            else:
                loss = per_graph_mse_loss(pred, y_norm, batch.batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs
        epoch_loss /= max(len(train_loader.dataset), 1)

        train_metrics = evaluate(
            model,
            train_loader,
            device,
            target_mean=target_mean,
            target_std=target_std,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            target_mean=target_mean,
            target_std=target_std,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            target_mean=target_mean,
            target_std=target_std,
        )

        train_score = metric_value(train_metrics, args.metric)
        val_score = metric_value(val_metrics, args.metric)
        test_score = metric_value(test_metrics, args.metric)
        train_curve.append(train_score)
        val_curve.append(val_score)
        test_curve.append(test_score)

        if val_score < best_val:
            best_val = val_score
            best_test = test_score

        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | "
            f"train {args.metric.upper()} {train_score:.4f} | "
            f"val {args.metric.upper()} {val_score:.4f} | "
            f"test {args.metric.upper()} {test_score:.4f} | "
            f"lr {current_lr:.1e} | "
            f"time {dt:.1f}s"
        )

        scheduler.step(val_score)

    return {
        "train_curve": train_curve,
        "val_curve": val_curve,
        "test_curve": test_curve,
        "final_test": test_curve[-1] if test_curve else 0.0,
        "best_val": best_val,
        "best_test": best_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train base model on all ablation datasets."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="pyg_dataset.pt",
        help="Full dataset path to include as baseline.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also train on the full dataset baseline.",
    )
    parser.add_argument(
        "--only-baseline",
        action="store_true",
        help="Train only the baseline dataset and skip ablation datasets.",
    )
    parser.add_argument("--datasets-dir", type=str, default="datasets")
    parser.add_argument("--pattern", type=str, default="pyg_dataset_no_*.pt")
    parser.add_argument("--output", type=str, default="ablation_results.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-batch-norm", action="store_true")
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_dir = os.path.abspath(args.datasets_dir)
    files = sorted(glob(os.path.join(datasets_dir, args.pattern)))
    if not args.only_baseline and not files:
        raise SystemExit(f"No datasets found in {datasets_dir} with pattern {args.pattern}.")

    results: Dict[str, Dict[str, List[float] | float]] = {}
    if args.include_baseline or args.only_baseline:
        set_seed(args.seed)
        baseline_path = os.path.abspath(args.baseline)
        if not os.path.exists(baseline_path):
            raise SystemExit(f"Baseline dataset not found: {baseline_path}")
        print("\n=== Training on baseline ===")
        baseline_dataset = torch.load(baseline_path, weights_only=False)
        results["baseline"] = train_base_model(baseline_dataset, args, device)

    if not args.only_baseline:
        for path in files:
            set_seed(args.seed)
            name = os.path.splitext(os.path.basename(path))[0]
            print(f"\n=== Training on {name} ===")
            dataset = torch.load(path, weights_only=False)
            results[name] = train_base_model(dataset, args, device)

    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric": args.metric,
                "epochs": args.epochs,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
