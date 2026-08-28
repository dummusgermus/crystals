"""Train CGCNN on totals datasets: atom-only loss vs atom + net-energy loss.

Uses grouped 70/15/15 split (train_single.grouped_split_indices).
Builds datasets via build_residual_datasets_with_totals.py if missing.

Example::

    python train_cgcnn_extensive_comparison.py --epochs 300
    python train_cgcnn_extensive_comparison.py --skip-build --epochs 300 --plot
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.loader import DataLoader

from gnn_models import build_gated_model_from_dataset
from train_single import (
    TotalLossConfig,
    combine_atom_total_loss,
    ensure_graph_delta_field,
    evaluate_with_total_energy,
    grouped_split_indices,
    metric_value,
    per_graph_mae_loss,
    per_graph_mse_loss,
    per_graph_total_loss_weights,
    per_graph_total_scaled_loss,
    per_graph_total_scaled_mse_loss,
    select_total_targets,
    set_seed,
    summarize_split,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "point": os.path.join(ROOT, "adv_datasets", "cycle34_residual_totals_dataset.pt"),
    "planar": os.path.join(
        ROOT, "planar_pyg_dataset_residual_c14c15_totals.pt"
    ),
}

DEFAULT_JSON = os.path.join(ROOT, "cgcnn_extensive_comparison_curves.json")
DEFAULT_PLOT = os.path.join(ROOT, "cgcnn_extensive_comparison_curves.png")

SWEEP_LAMBDAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SWEEP_JSON = os.path.join(ROOT, "cgcnn_lambda_sweep_curves.json")
SWEEP_SUMMARY_JSON = os.path.join(ROOT, "cgcnn_lambda_sweep_summary.json")

DEFAULT_CONFIG = dict(
    hidden_dim=128,
    num_layers=2,
    dropout=0.0,
    use_batch_norm=False,
    activation="silu",
    bidirectional=True,
    lr=2e-3,
    weight_decay=0.0,
    batch_size=8,
)


def build_datasets(force: bool = False) -> None:
    cmd = [sys.executable, os.path.join(ROOT, "build_residual_datasets_with_totals.py")]
    if force:
        cmd.append("--force")
    print("Building totals datasets …")
    subprocess.run(cmd, check=True, cwd=ROOT)


def _train_one(
    *,
    domain: str,
    dataset,
    device: torch.device,
    epochs: int,
    metric: str,
    seed: int,
    config: Dict,
    use_total_loss: bool,
    lambda_tot: float,
    run_key: Optional[str] = None,
    checkpoint_metric: str = "mae",
    total_loss_config: Optional[TotalLossConfig] = None,
    legacy_total_loss: bool = False,
) -> Dict:
    loss_tag = "atom_plus_total" if use_total_loss else "atom_only"
    name = run_key or (
        f"{domain}_lambda_{lambda_tot:g}" if use_total_loss else f"{domain}_atom_only"
    )
    set_seed(seed)

    ensure_graph_delta_field(dataset)

    train_idx, val_idx, test_idx = grouped_split_indices(dataset, seed)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    print(f"\n=== {name} | train={len(train_set)} val={len(val_set)} test={len(test_set)} ===")
    summarize_split(f"[{name}] Train", train_set)
    summarize_split(f"[{name}] Val", val_set)
    summarize_split(f"[{name}] Test", test_set)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    train_targets = torch.cat([d.y for d in train_set], dim=0).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    model = build_gated_model_from_dataset(
        dataset,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_batch_norm=config["use_batch_norm"],
        activation=config["activation"],
        bidirectional=config["bidirectional"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    curves = {
        "train_mae": [],
        "val_mae": [],
        "test_mae": [],
        "train_r_tot_median": [],
        "val_r_tot_median": [],
        "test_r_tot_median": [],
        "train_r_tot_mean": [],
        "val_r_tot_mean": [],
        "test_r_tot_mean": [],
        "train_abs_total_err_eV": [],
        "val_abs_total_err_eV": [],
        "test_abs_total_err_eV": [],
    }
    best_val = float("inf")
    best_state = None
    ckpt_metric = checkpoint_metric.lower()
    if ckpt_metric not in {"mae", "r_tot"}:
        raise ValueError("checkpoint_metric must be 'mae' or 'r_tot'")

    eval_target_mode = "full"
    if use_total_loss and total_loss_config is not None and not legacy_total_loss:
        eval_target_mode = total_loss_config.target_mode
    elif use_total_loss and legacy_total_loss:
        eval_target_mode = "full"
    else:
        eval_target_mode = "graph"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            y_norm = (batch.y - target_mean) / target_std
            if metric == "mae":
                atom_loss = per_graph_mae_loss(pred, y_norm, batch.batch)
            else:
                atom_loss = per_graph_mse_loss(pred, y_norm, batch.batch)

            if use_total_loss:
                pred_denorm = pred * target_std + target_mean
                if legacy_total_loss or total_loss_config is None:
                    tot_loss = per_graph_total_scaled_mse_loss(
                        pred_denorm, batch.delta_total_eV, batch.batch
                    )
                    loss = atom_loss + lambda_tot * tot_loss
                else:
                    targets = select_total_targets(
                        batch, target_mode=total_loss_config.target_mode
                    )
                    weights = per_graph_total_loss_weights(
                        batch,
                        targets,
                        max_delta_eV=total_loss_config.outlier_max_delta_eV,
                        max_mismatch_eV=total_loss_config.outlier_max_mismatch_eV,
                    )
                    tot_loss = per_graph_total_scaled_loss(
                        pred_denorm,
                        targets,
                        batch.batch,
                        scale_eps=total_loss_config.scale_eps,
                        loss_type=total_loss_config.loss_type,
                        huber_delta=total_loss_config.huber_delta,
                        weights=weights,
                    )
                    loss = combine_atom_total_loss(
                        atom_loss,
                        tot_loss,
                        lambda_tot=total_loss_config.lambda_tot,
                        balance_losses=total_loss_config.balance_losses,
                    )
            else:
                loss = atom_loss
            loss.backward()
            optimizer.step()

        train_atom, train_tot = evaluate_with_total_energy(
            model,
            train_loader,
            device,
            target_mean,
            target_std,
            total_target_mode=eval_target_mode,
        )
        val_atom, val_tot = evaluate_with_total_energy(
            model,
            val_loader,
            device,
            target_mean,
            target_std,
            total_target_mode=eval_target_mode,
        )
        test_atom, test_tot = evaluate_with_total_energy(
            model,
            test_loader,
            device,
            target_mean,
            target_std,
            total_target_mode=eval_target_mode,
        )

        curves["train_mae"].append(metric_value(train_atom, metric))
        curves["val_mae"].append(metric_value(val_atom, metric))
        curves["test_mae"].append(metric_value(test_atom, metric))
        curves["train_r_tot_median"].append(train_tot.median_rel_total_err_pct)
        curves["val_r_tot_median"].append(val_tot.median_rel_total_err_pct)
        curves["test_r_tot_median"].append(test_tot.median_rel_total_err_pct)
        curves["train_r_tot_mean"].append(train_tot.mean_rel_total_err_pct)
        curves["val_r_tot_mean"].append(val_tot.mean_rel_total_err_pct)
        curves["test_r_tot_mean"].append(test_tot.mean_rel_total_err_pct)
        curves["train_abs_total_err_eV"].append(train_tot.mean_abs_total_err_eV)
        curves["val_abs_total_err_eV"].append(val_tot.mean_abs_total_err_eV)
        curves["test_abs_total_err_eV"].append(test_tot.mean_abs_total_err_eV)

        scheduler.step(curves["val_mae"][-1])

        if ckpt_metric == "r_tot":
            score = curves["val_r_tot_median"][-1]
        else:
            score = curves["val_mae"][-1]
        if score < best_val:
            best_val = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            dt = time.time() - t0
            print(
                f"[{name}] ep {epoch:03d} | "
                f"val MAE {curves['val_mae'][-1]:.4f} | "
                f"val |dE| {curves['val_abs_total_err_eV'][-1]:.4f} eV | "
                f"val R_tot med {curves['val_r_tot_median'][-1]:.1f}% | "
                f"test R_tot med {curves['test_r_tot_median'][-1]:.1f}% | "
                f"{dt:.1f}s",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    _, final_val_tot = evaluate_with_total_energy(
        model,
        val_loader,
        device,
        target_mean,
        target_std,
        total_target_mode=eval_target_mode,
    )
    _, final_test_tot = evaluate_with_total_energy(
        model,
        test_loader,
        device,
        target_mean,
        target_std,
        total_target_mode=eval_target_mode,
    )

    result = {
        **curves,
        "run_key": name,
        "domain": domain,
        "loss_mode": loss_tag,
        "use_total_loss": use_total_loss,
        "lambda_tot": lambda_tot if use_total_loss else 0.0,
        "checkpoint_metric": ckpt_metric,
        "best_val_score": best_val,
        "best_val_mae": min(curves["val_mae"]) if curves["val_mae"] else float("nan"),
        "best_val_r_tot_median": min(curves["val_r_tot_median"])
        if curves["val_r_tot_median"]
        else float("nan"),
        "best_val_r_tot_mean": min(curves["val_r_tot_mean"])
        if curves["val_r_tot_mean"]
        else float("nan"),
        "best_test_mae": min(curves["test_mae"]) if curves["test_mae"] else float("nan"),
        "best_test_r_tot_median": min(curves["test_r_tot_median"])
        if curves["test_r_tot_median"]
        else float("nan"),
        "best_test_r_tot_mean": min(curves["test_r_tot_mean"])
        if curves["test_r_tot_mean"]
        else float("nan"),
        "final_val_mae": curves["val_mae"][-1],
        "final_val_r_tot_median": final_val_tot.median_rel_total_err_pct,
        "final_test_mae": curves["test_mae"][-1],
        "final_test_r_tot_median": final_test_tot.median_rel_total_err_pct,
        "final_val_abs_total_err_eV": curves["val_abs_total_err_eV"][-1],
        "final_test_abs_total_err_eV": curves["test_abs_total_err_eV"][-1],
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "total_target_mode": eval_target_mode,
        "legacy_total_loss": legacy_total_loss,
    }
    if total_loss_config is not None:
        result["total_loss_config"] = total_loss_config.__dict__
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CGCNN atom-only vs atom+total-energy loss comparison."
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--lambda-tot", type=float, default=1.0)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output-json", default=DEFAULT_JSON)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-output", default=DEFAULT_PLOT)
    args = parser.parse_args()

    if not args.skip_build:
        build_datasets(force=args.force_build)
    if args.build_only:
        print("Build-only done.")
        return

    for path in DATASETS.values():
        if not os.path.isfile(path):
            raise SystemExit(f"Missing dataset: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results: Dict[str, Dict] = {}
    for domain, path in DATASETS.items():
        dataset = torch.load(path, weights_only=False)
        if not hasattr(dataset[0], "delta_total_eV"):
            raise SystemExit(f"{path} lacks delta_total_eV; run with --force-build")

        for use_total in (False, True):
            tag = f"{domain}_{'atom_plus_total' if use_total else 'atom_only'}"
            results[tag] = _train_one(
                domain=domain,
                dataset=dataset,
                device=device,
                epochs=args.epochs,
                metric=args.metric,
                seed=args.seed,
                config=DEFAULT_CONFIG,
                use_total_loss=use_total,
                lambda_tot=args.lambda_tot,
                legacy_total_loss=use_total,
            )
            results[tag]["dataset_path"] = path

    payload = {
        "metric": args.metric,
        "epochs": args.epochs,
        "seed": args.seed,
        "split": "grouped 70/15/15",
        "lambda_tot": args.lambda_tot,
        "config": DEFAULT_CONFIG,
        "runs": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved curves to {args.output_json}")

    if args.plot:
        plot_script = os.path.join(ROOT, "plot_cgcnn_extensive_comparison.py")
        subprocess.run(
            [
                sys.executable,
                plot_script,
                "--input",
                args.output_json,
                "--output",
                args.plot_output,
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
