from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_model_from_dataset, build_transformer_model_from_dataset
from train_single import (
    evaluate,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    set_seed,
    summarize_split,
)


def train_and_collect_test_curve(
    model_name: str,
    model,
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
) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )
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

        val_metrics = evaluate(
            model,
            val_loader,
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
        test_score = metric_value(test_metrics, metric)
        val_score = metric_value(val_metrics, metric)
        test_curve.append(test_score)
        scheduler.step(val_score)

        dt = time.time() - t0
        print(
            f"[{model_name}] Epoch {epoch:03d} | "
            f"val {metric.upper()} {val_score:.4f} | "
            f"test {metric.upper()} {test_score:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e} | "
            f"time {dt:.1f}s"
        )

    return test_curve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train base and transformer models and store per-epoch test error in JSON."
    )
    parser.add_argument("--dataset", type=str, default="pyg_dataset.pt")
    parser.add_argument("--output", type=str, default="base_transformer_test_curves.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--transformer-batch-size",
        type=int,
        default=1,
        help="Batch size for transformer. Keep small: triangular attention is memory intensive.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-batch-norm", action="store_true")
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--seed", type=int, default=42)
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

    train_loader_base = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader_base = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader_base = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    train_loader_transformer = DataLoader(
        train_set, batch_size=args.transformer_batch_size, shuffle=True
    )
    val_loader_transformer = DataLoader(
        val_set, batch_size=args.transformer_batch_size, shuffle=False
    )
    test_loader_transformer = DataLoader(
        test_set, batch_size=args.transformer_batch_size, shuffle=False
    )

    train_targets = torch.cat([data.y for data in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    base_model = build_model_from_dataset(
        dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        activation=args.activation,
    ).to(device)
    transformer_model = build_transformer_model_from_dataset(
        dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        activation=args.activation,
    ).to(device)

    curves: Dict[str, List[float]] = {}
    print(
        f"Base batch size: {args.batch_size} | "
        f"Transformer batch size: {args.transformer_batch_size}"
    )
    curves["transformer"] = train_and_collect_test_curve(
        model_name="transformer",
        model=transformer_model,
        train_loader=train_loader_transformer,
        val_loader=val_loader_transformer,
        test_loader=test_loader_transformer,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        metric=args.metric,
    )
    curves["base"] = train_and_collect_test_curve(
        model_name="base",
        model=base_model,
        train_loader=train_loader_base,
        val_loader=val_loader_base,
        test_loader=test_loader_base,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        metric=args.metric,
    )

    payload = {
        "dataset": args.dataset,
        "metric": args.metric,
        "epochs": args.epochs,
        "seed": args.seed,
        "base_test_curve": curves["base"],
        "transformer_test_curve": curves["transformer"],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved test curves to {args.output}")


if __name__ == "__main__":
    main()
