from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_model_from_dataset


@dataclass
class Metrics:
    mae: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _per_graph_reduction(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    sums = torch.zeros(num_graphs, device=values.device)
    counts = torch.zeros(num_graphs, device=values.device)
    sums.index_add_(0, batch, values)
    counts.index_add_(0, batch, torch.ones_like(values))
    return sums / counts.clamp(min=1.0)


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


def compute_mae(pred: torch.Tensor, target: torch.Tensor, batch: torch.Tensor) -> Metrics:
    pred = pred.view(-1)
    target = target.view(-1)
    batch = batch.view(-1)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return Metrics(mae=0.0)
    abs_err = torch.abs(pred - target)
    per_graph_mae = _per_graph_reduction(abs_err, batch, num_graphs)
    return Metrics(mae=per_graph_mae.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train base model and save checkpoint.")
    parser.add_argument("--dataset", type=str, default="pyg_dataset.pt")
    parser.add_argument("--output", type=str, default="base_model.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-batch-norm", action="store_true")
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(args.dataset, weights_only=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train_targets = torch.cat([data.y for data in dataset], dim=0).view(-1)
    target_mean = train_targets.mean()
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

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

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            y_norm = (batch.y - target_mean) / target_std
            loss = per_graph_mae_loss(pred, y_norm, batch.batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= max(len(loader.dataset), 1)

        # Report MAE in original units
        model.eval()
        with torch.no_grad():
            preds = []
            targets = []
            batches = []
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch) * target_std + target_mean
                preds.append(pred)
                targets.append(batch.y)
                batches.append(batch.batch)
            pred = torch.cat(preds, dim=0)
            target = torch.cat(targets, dim=0)
            batch_idx = torch.cat(batches, dim=0)
            metrics = compute_mae(pred, target, batch_idx)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | "
            f"train MAE {metrics.mae:.4f} | "
            f"time {dt:.1f}s"
        )

    checkpoint: Dict = {
        "model_state": model.state_dict(),
        "config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "use_batch_norm": args.use_batch_norm,
            "activation": args.activation,
        },
        "target_mean": float(target_mean.detach().cpu().item()),
        "target_std": float(target_std.detach().cpu().item()),
        "metrics": asdict(metrics),
    }
    torch.save(checkpoint, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
