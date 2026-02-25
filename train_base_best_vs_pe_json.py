from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict, List

import torch
from torch_geometric.data import Data
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


class CosineWithWarmupLR:
    """Learning-rate schedule used by the PE transformer run."""

    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        lr: float,
        lr_decay_iters: int,
        min_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr = lr
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

    def __call__(self, epoch: int) -> None:
        lr = self._get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_iters:
            return self.lr * epoch / max(self.warmup_iters, 1)
        if epoch > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (epoch - self.warmup_iters) / max(
            self.lr_decay_iters - self.warmup_iters, 1
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)


def _token_index_for_graph(num_nodes: int, device: torch.device) -> torch.Tensor:
    if num_nodes <= 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    row = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    col = torch.arange(num_nodes, device=device).repeat(num_nodes)
    return torch.stack([row, col], dim=0)


def _induced_subgraph_by_nodes(data: Data, keep_idx: torch.Tensor) -> Data:
    keep_idx = keep_idx.to(dtype=torch.long, device=data.x.device)
    keep_idx = torch.unique(keep_idx, sorted=True)
    node_map = torch.full((data.num_nodes,), -1, dtype=torch.long, device=data.x.device)
    node_map[keep_idx] = torch.arange(keep_idx.numel(), device=data.x.device)

    src = data.edge_index[0]
    dst = data.edge_index[1]
    edge_mask = (node_map[src] >= 0) & (node_map[dst] >= 0)
    new_edge_index = node_map[data.edge_index[:, edge_mask]]

    new_data = data.clone()
    new_data.x = data.x[keep_idx]
    new_data.y = data.y[keep_idx]
    if getattr(data, "pos", None) is not None:
        new_data.pos = data.pos[keep_idx]
    new_data.edge_index = new_edge_index
    if getattr(data, "edge_attr", None) is not None:
        new_data.edge_attr = data.edge_attr[edge_mask]
    new_data.token_index = _token_index_for_graph(new_data.num_nodes, new_data.x.device)
    return new_data


def _prepare_transformer_dataset(dataset: List[Data], max_nodes: int) -> List[Data]:
    processed: List[Data] = []
    use_cap = max_nodes > 0

    for data in dataset:
        d = data.clone()
        if use_cap and d.num_nodes > max_nodes:
            # Column index 3 is distance-to-defect in graph_maker.py.
            dist = d.x[:, 3]
            keep_idx = torch.topk(dist, k=max_nodes, largest=False).indices
            d = _induced_subgraph_by_nodes(d, keep_idx)
        else:
            d.token_index = _token_index_for_graph(d.num_nodes, d.x.device)
        processed.append(d)
    return processed


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
    optimizer_name: str = "adam",
    scheduler_name: str = "plateau",
    warmup_iters: int = 0,
    min_lr: float = 0.0,
    gradient_norm: float | None = None,
) -> List[float]:
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_name = scheduler_name.lower()
    if scheduler_name == "cosine_warmup":
        scheduler = CosineWithWarmupLR(
            optimizer=optimizer,
            warmup_iters=warmup_iters,
            lr=lr,
            lr_decay_iters=epochs,
            min_lr=min_lr,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
        )
    test_curve: List[float] = []

    for epoch in range(1, epochs + 1):
        if scheduler_name == "cosine_warmup":
            scheduler(epoch - 1)
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
            if gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)
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
        if scheduler_name != "cosine_warmup":
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
        description=(
            "Train best base model and PE transformer model for 300 epochs and "
            "save per-epoch test error curves to JSON."
        )
    )
    parser.add_argument("--dataset", type=str, default="pyg_dataset.pt")
    parser.add_argument("--output", type=str, default="base_best_vs_pe_transformer_300.json")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--seed", type=int, default=42)

    # Best base config from previous sweep/compare runs.
    parser.add_argument("--base-hidden-dim", type=int, default=256)
    parser.add_argument("--base-num-layers", type=int, default=2)
    parser.add_argument("--base-dropout", type=float, default=0.0)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--base-weight-decay", type=float, default=0.0)
    parser.add_argument("--base-batch-size", type=int, default=16)
    parser.add_argument("--base-activation", type=str, default="gelu")
    parser.add_argument("--base-use-batch-norm", action="store_true")

    # PE transformer config (architecture from gnn_models transformer port).
    parser.add_argument("--transformer-hidden-dim", type=int, default=64)
    parser.add_argument("--transformer-num-layers", type=int, default=10)
    parser.add_argument("--transformer-num-heads", type=int, default=8)
    parser.add_argument("--transformer-attention-dropout", type=float, default=0.2)
    parser.add_argument("--transformer-ffn-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-activation", type=str, default="gelu")
    parser.add_argument("--transformer-lr", type=float, default=1e-3)
    parser.add_argument("--transformer-weight-decay", type=float, default=1e-5)
    parser.add_argument("--transformer-batch-size", type=int, default=32)
    parser.add_argument("--transformer-warmup-iters", type=int, default=50)
    parser.add_argument("--transformer-min-lr", type=float, default=0.0)
    parser.add_argument("--transformer-gradient-norm", type=float, default=1.0)
    parser.add_argument(
        "--transformer-max-nodes",
        type=int,
        default=0,
        help="0 keeps full graphs; >0 caps nodes/graph for transformer.",
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

    train_loader_base = DataLoader(train_set, batch_size=args.base_batch_size, shuffle=True)
    val_loader_base = DataLoader(val_set, batch_size=args.base_batch_size, shuffle=False)
    test_loader_base = DataLoader(test_set, batch_size=args.base_batch_size, shuffle=False)

    train_set_transformer = _prepare_transformer_dataset(
        train_set, max_nodes=args.transformer_max_nodes
    )
    val_set_transformer = _prepare_transformer_dataset(
        val_set, max_nodes=args.transformer_max_nodes
    )
    test_set_transformer = _prepare_transformer_dataset(
        test_set, max_nodes=args.transformer_max_nodes
    )
    train_loader_transformer = DataLoader(
        train_set_transformer, batch_size=args.transformer_batch_size, shuffle=True
    )
    val_loader_transformer = DataLoader(
        val_set_transformer, batch_size=args.transformer_batch_size, shuffle=False
    )
    test_loader_transformer = DataLoader(
        test_set_transformer, batch_size=args.transformer_batch_size, shuffle=False
    )

    train_targets = torch.cat([data.y for data in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    base_model = build_model_from_dataset(
        dataset,
        hidden_dim=args.base_hidden_dim,
        num_layers=args.base_num_layers,
        dropout=args.base_dropout,
        use_batch_norm=args.base_use_batch_norm,
        activation=args.base_activation,
    ).to(device)
    transformer_model = build_transformer_model_from_dataset(
        dataset,
        hidden_dim=args.transformer_hidden_dim,
        num_layers=args.transformer_num_layers,
        num_heads=args.transformer_num_heads,
        attention_dropout=args.transformer_attention_dropout,
        ffn_dropout=args.transformer_ffn_dropout,
        activation=args.transformer_activation,
    ).to(device)

    curves: Dict[str, List[float]] = {}
    curves["pe_transformer"] = train_and_collect_test_curve(
        model_name="pe_transformer",
        model=transformer_model,
        train_loader=train_loader_transformer,
        val_loader=val_loader_transformer,
        test_loader=test_loader_transformer,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        epochs=args.epochs,
        lr=args.transformer_lr,
        weight_decay=args.transformer_weight_decay,
        metric=args.metric,
        optimizer_name="adamw",
        scheduler_name="cosine_warmup",
        warmup_iters=args.transformer_warmup_iters,
        min_lr=args.transformer_min_lr,
        gradient_norm=args.transformer_gradient_norm,
    )
    curves["base_best"] = train_and_collect_test_curve(
        model_name="base_best",
        model=base_model,
        train_loader=train_loader_base,
        val_loader=val_loader_base,
        test_loader=test_loader_base,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        epochs=args.epochs,
        lr=args.base_lr,
        weight_decay=args.base_weight_decay,
        metric=args.metric,
        optimizer_name="adam",
        scheduler_name="plateau",
    )

    payload = {
        "dataset": args.dataset,
        "metric": args.metric,
        "epochs": args.epochs,
        "seed": args.seed,
        "base_best_config": {
            "hidden_dim": args.base_hidden_dim,
            "num_layers": args.base_num_layers,
            "dropout": args.base_dropout,
            "lr": args.base_lr,
            "weight_decay": args.base_weight_decay,
            "batch_size": args.base_batch_size,
            "activation": args.base_activation,
            "use_batch_norm": args.base_use_batch_norm,
        },
        "pe_transformer_config": {
            "hidden_dim": args.transformer_hidden_dim,
            "num_layers": args.transformer_num_layers,
            "num_heads": args.transformer_num_heads,
            "attention_dropout": args.transformer_attention_dropout,
            "ffn_dropout": args.transformer_ffn_dropout,
            "activation": args.transformer_activation,
            "lr": args.transformer_lr,
            "weight_decay": args.transformer_weight_decay,
            "batch_size": args.transformer_batch_size,
            "warmup_iters": args.transformer_warmup_iters,
            "min_lr": args.transformer_min_lr,
            "gradient_norm": args.transformer_gradient_norm,
            "max_nodes": args.transformer_max_nodes,
        },
        "base_best_test_curve": curves["base_best"],
        "pe_transformer_test_curve": curves["pe_transformer"],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved test curves to {args.output}")


if __name__ == "__main__":
    main()
