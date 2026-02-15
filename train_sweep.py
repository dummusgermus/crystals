from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from gnn_models import (
    build_model_from_dataset,
    build_hadamard_model_from_dataset,
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


def to_device(batch, device: torch.device):
    return batch.to(device)


def _per_graph_reduction(
    values: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    sums = torch.zeros(num_graphs, device=values.device)
    counts = torch.zeros(num_graphs, device=values.device)
    sums.index_add_(0, batch, values)
    counts.index_add_(0, batch, torch.ones_like(values))
    return sums / counts.clamp(min=1.0)


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    total_mse = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            pred = model(batch)
            num_graphs = batch.num_graphs
            m = compute_metrics(pred, batch.y, batch.batch)
            total_mse += m.mse * num_graphs
            total_rmse += m.rmse * num_graphs
            total_mae += m.mae * num_graphs
            total_graphs += num_graphs
    if total_graphs == 0:
        return Metrics(mse=0.0, rmse=0.0, mae=0.0)
    return Metrics(
        mse=total_mse / total_graphs,
        rmse=total_rmse / total_graphs,
        mae=total_mae / total_graphs,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = per_graph_mse_loss(pred, batch.y, batch.batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / max(len(loader.dataset), 1)


def kfold_indices(n_samples: int, k_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return np.array_split(indices, k_folds)


def parse_list_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep GNN configs on pyg dataset.")
    parser.add_argument("--dataset", type=str, default="pyg_dataset.pt")
    parser.add_argument("--output", type=str, default="gnn_sweep_results.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--use-batch-norm", action="store_true")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--activations",
        type=str,
        default="silu,relu,gelu,tanh,elu,leaky_relu",
        help="Comma-separated activation names.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(args.dataset, weights_only=False)
    n_samples = len(dataset)
    folds = kfold_indices(n_samples, args.k_folds, args.seed)

    activations = parse_list_arg(args.activations)
    model_types = ["base", "hadamard", "transformer"]

    results: Dict = {
        "meta": {
            "dataset": args.dataset,
            "n_samples": n_samples,
            "k_folds": args.k_folds,
            "seed": args.seed,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparams": {
                "epochs": args.epochs,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "use_batch_norm": args.use_batch_norm,
            },
        },
        "runs": [],
    }

    for model_type in model_types:
        for activation in activations:
                fold_metrics = []
                for fold_idx in range(args.k_folds):
                    test_idx = folds[fold_idx]
                    val_idx = folds[(fold_idx + 1) % args.k_folds]
                    train_idx = np.hstack(
                        [folds[i] for i in range(args.k_folds) if i not in {fold_idx, (fold_idx + 1) % args.k_folds}]
                    )

                    train_set = [dataset[i] for i in train_idx]
                    val_set = [dataset[i] for i in val_idx]
                    test_set = [dataset[i] for i in test_idx]

                    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
                    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
                    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

                    if model_type == "base":
                        model = build_model_from_dataset(
                            dataset,
                            hidden_dim=args.hidden_dim,
                            num_layers=args.num_layers,
                            dropout=args.dropout,
                            use_batch_norm=args.use_batch_norm,
                            activation=activation,
                        )
                    else:
                        if model_type == "hadamard":
                            model = build_hadamard_model_from_dataset(
                                dataset,
                                hidden_dim=args.hidden_dim,
                                num_layers=args.num_layers,
                                dropout=args.dropout,
                                use_batch_norm=args.use_batch_norm,
                                activation=activation,
                            )
                        else:
                            model = build_transformer_model_from_dataset(
                                dataset,
                                hidden_dim=args.hidden_dim,
                                num_layers=args.num_layers,
                                num_heads=args.num_heads,
                                dropout=args.dropout,
                                activation=activation,
                            )

                    model = model.to(device)
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                    )
                    best_val = float("inf")
                    best_state = None
                    epochs_no_improve = 0

                    for _ in range(args.epochs):
                        train_one_epoch(model, train_loader, device, optimizer)
                        val_metrics = evaluate(model, val_loader, device)
                        if val_metrics.mse < best_val - args.min_delta:
                            best_val = val_metrics.mse
                            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= args.patience:
                                break

                    if best_state is not None:
                        model.load_state_dict(best_state)

                    test_metrics = evaluate(model, test_loader, device)
                    fold_metrics.append(
                        {
                            "fold": fold_idx,
                            "test": asdict(test_metrics),
                        }
                    )

                mse_values = [fm["test"]["mse"] for fm in fold_metrics]
                rmse_values = [fm["test"]["rmse"] for fm in fold_metrics]
                mae_values = [fm["test"]["mae"] for fm in fold_metrics]

                run_result = {
                    "model_type": model_type,
                    "activation": activation,
                    "folds": fold_metrics,
                    "mean": {
                        "mse": float(np.mean(mse_values)),
                        "rmse": float(np.mean(rmse_values)),
                        "mae": float(np.mean(mae_values)),
                    },
                    "std": {
                        "mse": float(np.std(mse_values)),
                        "rmse": float(np.std(rmse_values)),
                        "mae": float(np.std(mae_values)),
                    },
                }
                results["runs"].append(run_result)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
