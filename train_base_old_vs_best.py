from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import matplotlib.pyplot as plt
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


def train_curve(
    dataset,
    train_set,
    val_set,
    test_set,
    device: torch.device,
    metric: str,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    epochs: int,
    cfg: Dict,
) -> Dict[str, List[float]]:
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False)

    model = build_model_from_dataset(
        dataset,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_batch_norm=cfg["use_batch_norm"],
        activation=cfg["activation"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    train_curve_vals: List[float] = []
    val_curve_vals: List[float] = []
    test_curve_vals: List[float] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
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

        train_metrics = evaluate(
            model, train_loader, device, target_mean=target_mean, target_std=target_std
        )
        val_metrics = evaluate(
            model, val_loader, device, target_mean=target_mean, target_std=target_std
        )
        test_metrics = evaluate(
            model, test_loader, device, target_mean=target_mean, target_std=target_std
        )

        train_score = metric_value(train_metrics, metric)
        val_score = metric_value(val_metrics, metric)
        test_score = metric_value(test_metrics, metric)
        train_curve_vals.append(train_score)
        val_curve_vals.append(val_score)
        test_curve_vals.append(test_score)
        scheduler.step(val_score)

        dt = time.time() - t0
        print(
            f"[{cfg['name']}] Epoch {epoch:03d} | "
            f"train {metric.upper()} {train_score:.4f} | "
            f"val {metric.upper()} {val_score:.4f} | "
            f"test {metric.upper()} {test_score:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e} | "
            f"time {dt:.1f}s"
        )

    return {
        "train_curve": train_curve_vals,
        "val_curve": val_curve_vals,
        "test_curve": test_curve_vals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train base model with old vs best config and plot test curves."
    )
    parser.add_argument("--dataset", type=str, default="pyg_dataset.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=str,
        default="base_old_vs_best_200.json",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="base_old_vs_best_200.png",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(args.dataset, weights_only=False)

    train_idx, val_idx, test_idx = grouped_split_indices(dataset, args.seed)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    summarize_split("Train", train_set)
    summarize_split("Val", val_set)
    summarize_split("Test", test_set)

    train_targets = torch.cat([data.y for data in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    old_cfg = {
        "name": "old",
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.1,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "activation": "silu",
        "use_batch_norm": False,
    }
    best_cfg = {
        "name": "best",
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 16,
        "activation": "gelu",
        "use_batch_norm": False,
    }

    print("\n=== Training old config ===")
    old_curves = train_curve(
        dataset=dataset,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        device=device,
        metric=args.metric,
        target_mean=target_mean,
        target_std=target_std,
        epochs=args.epochs,
        cfg=old_cfg,
    )
    print("\n=== Training best config ===")
    best_curves = train_curve(
        dataset=dataset,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        device=device,
        metric=args.metric,
        target_mean=target_mean,
        target_std=target_std,
        epochs=args.epochs,
        cfg=best_cfg,
    )

    payload = {
        "dataset": args.dataset,
        "metric": args.metric,
        "epochs": args.epochs,
        "seed": args.seed,
        "old_config": old_cfg,
        "best_config": best_cfg,
        "old": old_curves,
        "best": best_curves,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON to {args.output_json}")

    plt.figure(figsize=(8, 5))
    plt.plot(old_curves["test_curve"], label=f"Old test {args.metric.upper()}")
    plt.plot(best_curves["test_curve"], label=f"Best test {args.metric.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel(args.metric.upper())
    plt.title(f"Base model: old vs best ({args.epochs} epochs)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    print(f"Saved plot to {args.output_plot}")


if __name__ == "__main__":
    main()
