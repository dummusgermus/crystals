from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
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


@dataclass
class SweepConfig:
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    activation: str
    use_batch_norm: bool


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_str_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_bool_list(value: str) -> List[bool]:
    out: List[bool] = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in {"true", "1", "yes", "y"}:
            out.append(True)
        elif token in {"false", "0", "no", "n"}:
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean token: {raw}")
    return out


def make_configs(args: argparse.Namespace) -> List[SweepConfig]:
    hidden_dims = parse_int_list(args.hidden_dims)
    num_layers_list = parse_int_list(args.num_layers_list)
    dropouts = parse_float_list(args.dropouts)
    lrs = parse_float_list(args.lrs)
    weight_decays = parse_float_list(args.weight_decays)
    batch_sizes = parse_int_list(args.batch_sizes)
    activations = parse_str_list(args.activations)
    use_batch_norm_list = parse_bool_list(args.use_batch_norm_list)

    configs: List[SweepConfig] = []
    for vals in itertools.product(
        hidden_dims,
        num_layers_list,
        dropouts,
        lrs,
        weight_decays,
        batch_sizes,
        activations,
        use_batch_norm_list,
    ):
        configs.append(
            SweepConfig(
                hidden_dim=vals[0],
                num_layers=vals[1],
                dropout=vals[2],
                lr=vals[3],
                weight_decay=vals[4],
                batch_size=vals[5],
                activation=vals[6],
                use_batch_norm=vals[7],
            )
        )
    return configs


def train_one_config(
    dataset,
    cfg: SweepConfig,
    train_set,
    val_set,
    test_set,
    device: torch.device,
    epochs: int,
    metric: str,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> Dict:
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    model = build_model_from_dataset(
        dataset,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_batch_norm=cfg.use_batch_norm,
        activation=cfg.activation,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    test_curve: List[float] = []
    val_curve: List[float] = []
    train_curve: List[float] = []
    epoch_times: List[float] = []

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
            model,
            train_loader,
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
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            target_mean=target_mean,
            target_std=target_std,
        )

        train_score = metric_value(train_metrics, metric)
        val_score = metric_value(val_metrics, metric)
        test_score = metric_value(test_metrics, metric)
        train_curve.append(train_score)
        val_curve.append(val_score)
        test_curve.append(test_score)
        scheduler.step(val_score)

        epoch_dt = time.time() - t0
        epoch_times.append(epoch_dt)

        print(
            f"[cfg bs={cfg.batch_size} hd={cfg.hidden_dim} nl={cfg.num_layers} act={cfg.activation} bn={cfg.use_batch_norm}] "
            f"Epoch {epoch:03d} | train {metric.upper()} {train_score:.4f} | "
            f"val {metric.upper()} {val_score:.4f} | test {metric.upper()} {test_score:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e} | time {epoch_dt:.1f}s"
        )

    best_idx = min(range(len(val_curve)), key=lambda i: val_curve[i]) if val_curve else 0
    return {
        "config": asdict(cfg),
        "train_curve": train_curve,
        "val_curve": val_curve,
        "test_curve": test_curve,
        "best_epoch": int(best_idx + 1),
        "best_val": float(val_curve[best_idx]) if val_curve else 0.0,
        "test_at_best_val": float(test_curve[best_idx]) if test_curve else 0.0,
        "final_test": float(test_curve[-1]) if test_curve else 0.0,
        "epoch_times_sec": epoch_times,
        "total_time_sec": float(sum(epoch_times)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Large hyperparameter sweep for base model with per-epoch test curves."
    )
    parser.add_argument("--dataset", type=str, default="pyg_dataset.pt")
    parser.add_argument("--output", type=str, default="base_hparam_sweep_curves.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run all combinations.")

    # Grid definitions (comma-separated lists)
    parser.add_argument("--hidden-dims", type=str, default="64,128,256")
    parser.add_argument("--num-layers-list", type=str, default="2,4")
    parser.add_argument("--dropouts", type=str, default="0.0,0.1")
    parser.add_argument("--lrs", type=str, default="1e-3,5e-4,1e-4")
    parser.add_argument("--weight-decays", type=str, default="0.0,1e-5")
    parser.add_argument("--batch-sizes", type=str, default="16,32")
    parser.add_argument("--activations", type=str, default="relu,silu,gelu")
    parser.add_argument("--use-batch-norm-list", type=str, default="false")
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

    configs = make_configs(args)
    if args.max_runs > 0:
        configs = configs[: args.max_runs]

    payload: Dict = {
        "meta": {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "metric": args.metric,
            "seed": args.seed,
            "device": str(device),
            "num_configs_total": len(make_configs(args)),
            "num_configs_run": len(configs),
            "split_sizes": {
                "train_graphs": len(train_set),
                "val_graphs": len(val_set),
                "test_graphs": len(test_set),
            },
            "grid": {
                "hidden_dims": parse_int_list(args.hidden_dims),
                "num_layers_list": parse_int_list(args.num_layers_list),
                "dropouts": parse_float_list(args.dropouts),
                "lrs": parse_float_list(args.lrs),
                "weight_decays": parse_float_list(args.weight_decays),
                "batch_sizes": parse_int_list(args.batch_sizes),
                "activations": parse_str_list(args.activations),
                "use_batch_norm_list": parse_bool_list(args.use_batch_norm_list),
            },
        },
        "runs": [],
    }

    t_global = time.time()
    for run_idx, cfg in enumerate(configs, start=1):
        set_seed(args.seed + run_idx)
        print(
            f"\n=== Run {run_idx}/{len(configs)} | "
            f"hd={cfg.hidden_dim}, nl={cfg.num_layers}, do={cfg.dropout}, "
            f"lr={cfg.lr}, wd={cfg.weight_decay}, bs={cfg.batch_size}, "
            f"act={cfg.activation}, bn={cfg.use_batch_norm} ==="
        )
        result = train_one_config(
            dataset=dataset,
            cfg=cfg,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            device=device,
            epochs=args.epochs,
            metric=args.metric,
            target_mean=target_mean,
            target_std=target_std,
        )
        payload["runs"].append(result)

        # Persist incrementally in case long sweeps get interrupted.
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved partial results to {args.output}")

    payload["meta"]["total_runtime_sec"] = float(time.time() - t_global)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved final sweep results to {args.output}")


if __name__ == "__main__":
    main()
