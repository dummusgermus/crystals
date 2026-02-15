from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from gnn_models import (
    build_hadamard_model_from_dataset,
    build_model_from_dataset,
    build_transformer_model_from_dataset,
)


@dataclass
class Metrics:
    mse: float
    rmse: float
    mae: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _per_graph_reduction(
    values: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    sums = torch.zeros(num_graphs, device=values.device)
    counts = torch.zeros(num_graphs, device=values.device)
    sums.index_add_(0, batch, values)
    counts.index_add_(0, batch, torch.ones_like(values))
    return sums / counts.clamp(min=1.0)


def per_graph_mse_loss(pred: torch.Tensor, target: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    batch = batch.view(-1)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return torch.tensor(0.0, device=pred.device)
    sq = (pred - target) ** 2
    per_graph_mse = _per_graph_reduction(sq, batch, num_graphs)
    return per_graph_mse.mean()


def per_graph_mae_loss(pred: torch.Tensor, target: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)
    batch = batch.view(-1)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return torch.tensor(0.0, device=pred.device)
    abs_err = torch.abs(pred - target)
    per_graph_mae = _per_graph_reduction(abs_err, batch, num_graphs)
    return per_graph_mae.mean()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, batch: torch.Tensor) -> Metrics:
    pred = pred.view(-1)
    target = target.view(-1)
    batch = batch.view(-1)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return Metrics(mse=0.0, rmse=0.0, mae=0.0)

    sq = (pred - target) ** 2
    abs_err = torch.abs(pred - target)
    per_graph_mse = _per_graph_reduction(sq, batch, num_graphs)
    per_graph_mae = _per_graph_reduction(abs_err, batch, num_graphs)
    mse = per_graph_mse.mean().item()
    rmse = math.sqrt(mse)
    mae = per_graph_mae.mean().item()
    return Metrics(mse=mse, rmse=rmse, mae=mae)


def metric_value(metrics: Metrics, metric: str) -> float:
    metric = metric.lower()
    if metric == "rmse":
        return metrics.rmse
    if metric == "mae":
        return metrics.mae
    if metric == "mse":
        return metrics.mse
    raise ValueError("Unsupported metric. Choose from: rmse, mae, mse.")


def evaluate(
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
            m = compute_metrics(pred_denorm, batch.y, batch.batch)
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


def split_indices(n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def _group_key(data) -> Tuple:
    parts = (
        getattr(data, "defect_id", None),
        getattr(data, "from_type", None),
        getattr(data, "to_type", None),
        getattr(data, "wyckoff", None),
    )
    if all(p is None for p in parts):
        return ("idx", id(data))
    return parts


def grouped_split_indices(dataset, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: Dict[Tuple, List[int]] = {}
    for idx, data in enumerate(dataset):
        key = _group_key(data)
        groups.setdefault(key, []).append(idx)

    rng = np.random.default_rng(seed)
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)

    n_samples = len(dataset)
    target_train = int(0.7 * n_samples)
    target_val = int(0.15 * n_samples)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for key in group_keys:
        group_indices = groups[key]
        if len(train_idx) < target_train:
            train_idx.extend(group_indices)
        elif len(val_idx) < target_val:
            val_idx.extend(group_indices)
        else:
            test_idx.extend(group_indices)

    if len(val_idx) == 0 or len(test_idx) == 0:
        return split_indices(n_samples, seed)

    return (
        np.array(train_idx, dtype=int),
        np.array(val_idx, dtype=int),
        np.array(test_idx, dtype=int),
    )


def summarize_split(name: str, subset) -> None:
    num_graphs = len(subset)
    if num_graphs == 0:
        print(f"{name} split: 0 graphs")
        return
    node_counts = [int(data.num_nodes) for data in subset]
    total_nodes = int(np.sum(node_counts))
    y_all = torch.cat([data.y for data in subset], dim=0).view(-1)
    y_mean = float(y_all.mean().item())
    y_std = float(y_all.std(unbiased=False).item())
    print(
        f"{name} split: graphs={num_graphs}, nodes={total_nodes}, "
        f"nodes/graph mean={np.mean(node_counts):.1f}, median={np.median(node_counts):.1f}, "
        f"y mean={y_mean:.4f}, y std={y_std:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Single training run for node PE prediction.")
    parser.add_argument("--dataset", type=str, default="pyg_dataset.pt")
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["base", "hadamard", "transformer", "both", "all"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--use-batch-norm", action="store_true")
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", type=str, default="train_test_curve.png")
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

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    train_targets = torch.cat([data.y for data in train_set], dim=0).view(-1)
    target_mean = train_targets.mean()
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    def build_model(model_name: str):
        if model_name == "base":
            return build_model_from_dataset(
                dataset,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_batch_norm=args.use_batch_norm,
                activation=args.activation,
            )
        if model_name == "transformer":
            return build_transformer_model_from_dataset(
                dataset,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                attention_dropout=args.dropout,
                ffn_dropout=args.dropout,
                activation=args.activation,
            )
        return build_hadamard_model_from_dataset(
            dataset,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_batch_norm=args.use_batch_norm,
            activation=args.activation,
        )

    def train_model(model_name: str) -> List[float]:
        model = build_model(model_name).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
        )
        test_curve: List[float] = []

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

            test_curve.append(metric_value(test_metrics, args.metric))

            dt = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[{model_name}] Epoch {epoch:03d} | "
                f"train {args.metric.upper()} {metric_value(train_metrics, args.metric):.4f} | "
                f"val {args.metric.upper()} {metric_value(val_metrics, args.metric):.4f} | "
                f"test {args.metric.upper()} {metric_value(test_metrics, args.metric):.4f} | "
                f"lr {current_lr:.1e} | "
                f"time {dt:.1f}s"
            )

            scheduler.step(metric_value(val_metrics, args.metric))

        return test_curve

    test_curves: Dict[str, List[float]] = {}
    if args.model in {"base", "both", "all"}:
        test_curves["base"] = train_model("base")
    if args.model in {"hadamard", "both", "all"}:
        test_curves["hadamard"] = train_model("hadamard")
    if args.model in {"transformer", "all"}:
        test_curves["transformer"] = train_model("transformer")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it and rerun.") from exc

    plt.figure(figsize=(8, 5))
    for name, curve in test_curves.items():
        if name == "base":
            label = f"Base {args.metric.upper()}"
        elif name == "hadamard":
            label = f"Hadamard {args.metric.upper()}"
        else:
            label = f"Transformer {args.metric.upper()}"
        plt.plot(curve, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(args.metric.upper())
    plt.title(f"Test {args.metric.upper()} over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
