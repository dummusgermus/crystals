"""Train a Graph Transformer on absolute and residual point/planar datasets.

Uses the scalable :class:`GraphTransformerNodeRegressor` (TransformerEncoder +
edge attention bias) adapted from the PE / GDT transformer experiments under
``tests/``. Same split / metric protocol as CGCNN so results are comparable.

Default architecture is ``graph`` (CPU-friendly). Pass ``--architecture gts``
to use the dense triangular GTS transformer (prefer GPU).

Example::

    python train_transformer_absolute_residual.py --epochs 300
    python train_transformer_absolute_residual.py --epochs 300 --datasets residual
    python train_transformer_absolute_residual.py --architecture gts --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from torch_geometric.loader import DataLoader

from gnn_models import (
    build_graph_transformer_from_dataset,
    build_transformer_model_from_dataset,
)
from train_single import (
    evaluate,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    set_seed,
    summarize_split,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS: OrderedDict[str, Dict[str, str]] = OrderedDict(
    [
        (
            "defect",
            {
                "path": os.path.join(ROOT, "adv_datasets", "cycle34_dataset.pt"),
                "target_mode": "absolute",
            },
        ),
        (
            "planar_c14c15",
            {
                "path": os.path.join(ROOT, "planar_pyg_dataset_c14c15.pt"),
                "target_mode": "absolute",
            },
        ),
        (
            "defect_residual",
            {
                "path": os.path.join(
                    ROOT, "adv_datasets", "cycle34_residual_dataset.pt"
                ),
                "target_mode": "residual",
            },
        ),
        (
            "planar_residual_c14c15",
            {
                "path": os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15.pt"),
                "target_mode": "residual",
            },
        ),
    ]
)

CURVES_JSON = os.path.join(ROOT, "transformer_absolute_residual_curves.json")

# Scalable Graph Transformer defaults (CPU-friendly, comparable capacity).
GRAPH_CONFIG = dict(
    hidden_dim=128,
    num_layers=4,
    num_heads=4,
    dropout=0.1,
    attention_dropout=0.1,
    activation="gelu",
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=8,
    warmup_iters=20,
    min_lr=0.0,
    gradient_norm=1.0,
    scheduler="cosine_warmup",
    plateau_patience=16,
)

# Original GTS recipe from tests/run_train_base_transformer.slurm.
GTS_CONFIG = dict(
    hidden_dim=64,
    num_layers=10,
    num_heads=8,
    attention_dropout=0.2,
    ffn_dropout=0.0,
    activation="gelu",
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=1,
    warmup_iters=50,
    min_lr=0.0,
    gradient_norm=1.0,
)


class CosineWithWarmupLR:
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
        # epoch is 0-based; use epoch+1 so the first step is not lr=0.
        step = epoch + 1
        if step <= self.warmup_iters:
            return self.lr * step / max(self.warmup_iters, 1)
        if step > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (step - self.warmup_iters) / max(
            self.lr_decay_iters - self.warmup_iters, 1
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)


def _select_datasets(subset: str) -> OrderedDict[str, Dict[str, str]]:
    if subset == "all":
        return DATASETS
    if subset == "absolute":
        keys = ("defect", "planar_c14c15")
    elif subset == "residual":
        keys = ("defect_residual", "planar_residual_c14c15")
    elif subset == "point":
        keys = ("defect", "defect_residual")
    elif subset == "planar":
        keys = ("planar_c14c15", "planar_residual_c14c15")
    else:
        raise ValueError(f"Unknown dataset subset: {subset}")
    return OrderedDict((k, DATASETS[k]) for k in keys)


def _build_model(architecture: str, dataset, config: Dict):
    if architecture == "graph":
        return build_graph_transformer_from_dataset(
            dataset,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            attention_dropout=config["attention_dropout"],
            activation=config["activation"],
        )
    if architecture == "gts":
        return build_transformer_model_from_dataset(
            dataset,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            attention_dropout=config["attention_dropout"],
            ffn_dropout=config.get("ffn_dropout", 0.0),
            activation=config["activation"],
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def _mean_last_n(values: List[float], n: int) -> float:
    if not values:
        return float("inf")
    window = values[-n:] if n > 0 else values
    return float(sum(window) / len(window))


def _std_last_n(values: List[float], n: int) -> float:
    if not values:
        return float("inf")
    window = values[-n:] if n > 0 else values
    if len(window) < 2:
        return 0.0
    mean = sum(window) / len(window)
    var = sum((v - mean) ** 2 for v in window) / len(window)
    return float(math.sqrt(var))


def _train_one(
    name: str,
    dataset,
    device: torch.device,
    epochs: int,
    metric: str,
    seed: int,
    architecture: str,
    config: Dict,
    target_mode: str,
    checkpoint_path: Optional[str] = None,
    score_last_n: int = 50,
    quiet_every: int = 10,
) -> Dict:
    set_seed(seed)

    train_idx, val_idx, test_idx = grouped_split_indices(dataset, seed)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    summarize_split(f"[{name}] Train", train_set)
    summarize_split(f"[{name}] Val", val_set)
    summarize_split(f"[{name}] Test", test_set)

    bs = config["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

    train_targets = torch.cat([d.y for d in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    model = _build_model(architecture, dataset, config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] params={n_params:,} architecture={architecture}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler_name = str(config.get("scheduler", "cosine_warmup")).lower()
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=int(config.get("plateau_patience", 16)),
            min_lr=float(config.get("min_lr", 1e-6)),
        )
    else:
        scheduler = CosineWithWarmupLR(
            optimizer=optimizer,
            warmup_iters=config["warmup_iters"],
            lr=config["lr"],
            lr_decay_iters=epochs,
            min_lr=config["min_lr"],
        )

    train_curve: List[float] = []
    val_curve: List[float] = []
    test_curve: List[float] = []
    best_val = float("inf")
    best_state = None
    # Within the final score_last_n window, keep the lowest-val checkpoint.
    best_last_window_val = float("inf")
    best_last_window_state = None
    grad_clip = config.get("gradient_norm")

    for epoch in range(1, epochs + 1):
        if scheduler_name != "plateau":
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
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        train_m = evaluate(model, train_loader, device, target_mean, target_std)
        val_m = evaluate(model, val_loader, device, target_mean, target_std)
        test_m = evaluate(model, test_loader, device, target_mean, target_std)

        train_curve.append(metric_value(train_m, metric))
        val_curve.append(metric_value(val_m, metric))
        test_curve.append(metric_value(test_m, metric))

        if scheduler_name == "plateau":
            scheduler.step(val_curve[-1])

        if val_curve[-1] < best_val:
            best_val = val_curve[-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch > epochs - score_last_n and val_curve[-1] < best_last_window_val:
            best_last_window_val = val_curve[-1]
            best_last_window_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        if quiet_every > 0 and (
            epoch % quiet_every == 0 or epoch == 1 or epoch == epochs
        ):
            dt = time.time() - t0
            print(
                f"[{name}] Epoch {epoch:03d} | "
                f"train {metric.upper()} {train_curve[-1]:.4f} | "
                f"val {metric.upper()} {val_curve[-1]:.4f} | "
                f"test {metric.upper()} {test_curve[-1]:.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.1e} | {dt:.1f}s",
                flush=True,
            )

    assert best_state is not None
    save_state = best_last_window_state or best_state
    model.load_state_dict({k: v.to(device) for k, v in save_state.items()})
    final_val = evaluate(model, val_loader, device, target_mean, target_std)
    final_test = evaluate(model, test_loader, device, target_mean, target_std)

    if checkpoint_path:
        torch.save(
            {
                "model_state": save_state,
                "architecture": architecture,
                "config": config,
                "target_mean": float(target_mean.cpu()),
                "target_std": float(target_std.cpu()),
                "dataset": name,
                "score_last_n": score_last_n,
            },
            checkpoint_path,
        )
        print(f"[{name}] Saved checkpoint to {checkpoint_path}")

    return {
        "train": train_curve,
        "val": val_curve,
        "test": test_curve,
        "final_train": train_curve[-1],
        "final_val": val_curve[-1],
        "final_test": test_curve[-1],
        "best_val": metric_value(final_val, metric),
        "best_test": metric_value(final_test, metric),
        "last_n": score_last_n,
        "last_n_val_mean": _mean_last_n(val_curve, score_last_n),
        "last_n_test_mean": _mean_last_n(test_curve, score_last_n),
        "last_n_val_std": _std_last_n(val_curve, score_last_n),
        "last_n_test_std": _std_last_n(test_curve, score_last_n),
        "target_mode": target_mode,
        "num_params": n_params,
        "config": dict(config),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Graph Transformer on absolute/residual point & planar datasets."
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric", type=str, default="mae", choices=["mae", "rmse", "mse"]
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="graph",
        choices=["graph", "gts"],
        help="graph=TransformerEncoder+edge bias (default); gts=triangular GTS.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        choices=["all", "absolute", "residual", "point", "planar"],
    )
    parser.add_argument("--output-json", type=str, default=CURVES_JSON)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attention-dropout", type=float, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--gradient-norm", type=float, default=None)
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["cosine_warmup", "plateau"],
    )
    parser.add_argument(
        "--score-last-n",
        type=int,
        default=50,
        help="Report mean/std over the last N epoch MAEs (also used for ckpt pick).",
    )
    parser.add_argument("--config-json", type=str, default=None)
    args = parser.parse_args()

    config = dict(GRAPH_CONFIG if args.architecture == "graph" else GTS_CONFIG)
    config.setdefault("scheduler", "cosine_warmup")
    if args.config_json:
        with open(args.config_json, encoding="utf-8") as fh:
            loaded = json.load(fh)
        if "best_config" in loaded:
            config.update(loaded["best_config"])
        elif "config" in loaded:
            config.update(loaded["config"])
        else:
            config.update(loaded)
    for key, val in (
        ("batch_size", args.batch_size),
        ("hidden_dim", args.hidden_dim),
        ("num_layers", args.num_layers),
        ("num_heads", args.num_heads),
        ("lr", args.lr),
        ("weight_decay", args.weight_decay),
        ("dropout", args.dropout),
        ("attention_dropout", args.attention_dropout),
        ("warmup_iters", args.warmup_iters),
        ("min_lr", args.min_lr),
        ("gradient_norm", args.gradient_norm),
        ("scheduler", args.scheduler),
    ):
        if val is not None:
            config[key] = val

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {args.architecture}")
    print(f"Config: {config}")

    selected = _select_datasets(args.datasets)
    results: Dict[str, Dict] = {}

    for name, meta in selected.items():
        path = meta["path"]
        if not os.path.isfile(path):
            raise SystemExit(f"Dataset not found for {name}: {path}")
        print(f"\n=== Training {name} on {path} ===")
        dataset = torch.load(path, weights_only=False)
        ckpt = None
        if args.save_checkpoints:
            ckpt = os.path.join(ROOT, f"transformer_{args.architecture}_{name}_model.pt")
        curves = _train_one(
            name=name,
            dataset=dataset,
            device=device,
            epochs=args.epochs,
            metric=args.metric,
            seed=args.seed,
            architecture=args.architecture,
            config=config,
            target_mode=meta["target_mode"],
            checkpoint_path=ckpt,
            score_last_n=args.score_last_n,
        )
        curves["dataset_path"] = path
        curves["num_graphs"] = len(dataset)
        results[name] = curves

        # Incremental save so long runs keep partial results.
        payload = {
            "metric": args.metric,
            "epochs": args.epochs,
            "seed": args.seed,
            "score_last_n": args.score_last_n,
            "model": (
                "GraphTransformer"
                if args.architecture == "graph"
                else "GTSTriangularTransformer"
            ),
            "architecture": args.architecture,
            "config": config,
            "datasets": results,
            "notes": (
                "Transformer trained with the same grouped splits / per-graph MAE "
                "protocol as CGCNN. Primary score = mean of last "
                f"{args.score_last_n} epoch val/test MAEs."
            ),
        }
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Updated {args.output_json} ({len(results)}/{len(selected)} datasets)")

    print(f"\nDone. Curves saved to {args.output_json}")
    print(f"\nSummary (mean last-{args.score_last_n} MAE):")
    for name, curves in results.items():
        print(
            f"  {name:28s} "
            f"val={curves['last_n_val_mean']:.6f}±{curves['last_n_val_std']:.6f}  "
            f"test={curves['last_n_test_mean']:.6f}±{curves['last_n_test_std']:.6f}"
        )


if __name__ == "__main__":
    main()
