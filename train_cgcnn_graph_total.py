"""Train CGCNN graph regressor on full-cell total residual (graph-level target).

Predicts delta_total_eV directly from the defect graph features — no per-atom
head, no summing node predictions.

Split: grouped 70/15/15 (train_single.grouped_split_indices).
Checkpoint: best validation median R_tot on |delta_true| >= 1e-6 eV.

Example::

    python train_cgcnn_graph_total.py --skip-build --epochs 300 --plot
    python train_cgcnn_graph_total.py --domain planar --epochs 500
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from gnn_models import build_gated_graph_model_from_dataset
from train_single import grouped_split_indices, set_seed

ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "point": os.path.join(ROOT, "adv_datasets", "cycle34_graph_total_target_dataset.pt"),
    "planar": os.path.join(
        ROOT, "planar_pyg_dataset_graph_total_target_c14c15.pt"
    ),
}

CURVES_JSON = os.path.join(ROOT, "cgcnn_graph_total_curves.json")
SUMMARY_JSON = os.path.join(ROOT, "cgcnn_graph_total_summary.json")
PLOT_PREFIX = os.path.join(ROOT, "cgcnn_graph_total")

MIN_DELTA_EV = 1e-6
DEFAULT_SEED = 42

DEFAULT_CONFIG = dict(
    hidden_dim=128,
    num_layers=2,
    dropout=0.0,
    use_batch_norm=False,
    activation="silu",
    bidirectional=True,
    aggregation="add",
    lr=2e-3,
    weight_decay=0.0,
    batch_size=8,
    huber_delta=1.0,
    scale_eps=1e-3,
    outlier_max_delta_eV=50.0,
)


@dataclass
class GraphTotalMetrics:
    mae_eV: float
    rmse_eV: float
    median_abs_err_eV: float
    mean_abs_err_eV: float
    median_r_tot_pct: float
    mean_r_tot_pct: float
    n_graphs: int
    n_rtot: int


def build_datasets(force: bool = False) -> None:
    cmd = [sys.executable, os.path.join(ROOT, "build_graph_total_target_datasets.py")]
    if force:
        cmd.append("--force")
    print("Building graph-total-target datasets …")
    subprocess.run(cmd, check=True, cwd=ROOT)


def graph_target(data) -> torch.Tensor:
    return data.y.view(1)


def summarize_graph_split(name: str, subset) -> None:
    if not subset:
        print(f"{name} split: 0 graphs")
        return
    targets = torch.stack([graph_target(d) for d in subset]).view(-1)
    nodes = [int(d.num_nodes) for d in subset]
    print(
        f"{name} split: graphs={len(subset)}, nodes={sum(nodes)}, "
        f"nodes/graph mean={np.mean(nodes):.1f}, "
        f"delta_total mean={targets.mean():.4f} eV, std={targets.std(unbiased=False):.4f} eV",
        flush=True,
    )


def _sample_weights(targets: torch.Tensor, max_delta_eV: float) -> torch.Tensor:
    w = torch.ones_like(targets)
    w[targets.abs() > max_delta_eV] = 0.25
    return w


def graph_total_loss(
    pred: torch.Tensor,
    targets: torch.Tensor,
    *,
    scale_eps: float,
    huber_delta: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Huber on scaled relative error (same spirit as global-v2 total term)."""
    pred = pred.view(-1)
    targets = targets.view(-1)
    scale = targets.abs().clamp(min=scale_eps)
    rel = (pred - targets) / scale
    loss = F.smooth_l1_loss(rel, torch.zeros_like(rel), beta=huber_delta, reduction="none")
    if weights is not None:
        loss = loss * weights.view(-1)
        return loss.sum() / weights.sum().clamp(min=1e-6)
    return loss.mean()


@torch.no_grad()
def evaluate_graph_total(
    model,
    loader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> GraphTotalMetrics:
    model.eval()
    abs_errs: List[float] = []
    sq_errs: List[float] = []
    rel_errs: List[float] = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).view(-1) * target_std + target_mean
        true = batch.y.view(-1)
        err = pred - true
        for e, t in zip(err.cpu().tolist(), true.cpu().tolist()):
            abs_errs.append(abs(e))
            sq_errs.append(e * e)
            if abs(t) >= MIN_DELTA_EV:
                rel_errs.append(100.0 * abs(e) / abs(t))

    if not abs_errs:
        return GraphTotalMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    return GraphTotalMetrics(
        mae_eV=float(np.mean(abs_errs)),
        rmse_eV=float(np.sqrt(np.mean(sq_errs))),
        median_abs_err_eV=float(np.median(abs_errs)),
        mean_abs_err_eV=float(np.mean(abs_errs)),
        median_r_tot_pct=float(np.median(rel_errs)) if rel_errs else float("nan"),
        mean_r_tot_pct=float(np.mean(rel_errs)) if rel_errs else float("nan"),
        n_graphs=len(abs_errs),
        n_rtot=len(rel_errs),
    )


def _train_one(
    *,
    domain: str,
    dataset,
    device: torch.device,
    epochs: int,
    seed: int,
    config: Dict,
) -> Dict:
    name = f"{domain}_graph_total"
    set_seed(seed)

    train_idx, val_idx, test_idx = grouped_split_indices(dataset, seed)
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    print(
        f"\n=== {name} | graph-level delta_total_eV | "
        f"train={len(train_set)} val={len(val_set)} test={len(test_set)} ===",
        flush=True,
    )
    summarize_graph_split(f"[{name}] Train", train_set)
    summarize_graph_split(f"[{name}] Val", val_set)
    summarize_graph_split(f"[{name}] Test", test_set)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    train_targets = torch.stack([graph_target(d) for d in train_set]).view(-1)
    target_mean = train_targets.mean().to(device)
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6).to(device)

    model = build_gated_graph_model_from_dataset(
        dataset,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_batch_norm=config["use_batch_norm"],
        activation=config["activation"],
        bidirectional=config["bidirectional"],
        aggregation=config["aggregation"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] params={n_params:,}", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    curves = {
        "train_mae_eV": [],
        "val_mae_eV": [],
        "test_mae_eV": [],
        "train_r_tot_median": [],
        "val_r_tot_median": [],
        "test_r_tot_median": [],
        "train_abs_err_eV": [],
        "val_abs_err_eV": [],
        "test_abs_err_eV": [],
    }
    best_val_rtot = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_norm = model(batch)
            pred = pred_norm.view(-1) * target_std + target_mean
            true = batch.y.view(-1)
            weights = _sample_weights(true, config["outlier_max_delta_eV"])
            loss = graph_total_loss(
                pred,
                true,
                scale_eps=config["scale_eps"],
                huber_delta=config["huber_delta"],
                weights=weights,
            )
            loss.backward()
            optimizer.step()

        train_m = evaluate_graph_total(model, train_loader, device, target_mean, target_std)
        val_m = evaluate_graph_total(model, val_loader, device, target_mean, target_std)
        test_m = evaluate_graph_total(model, test_loader, device, target_mean, target_std)

        curves["train_mae_eV"].append(train_m.mae_eV)
        curves["val_mae_eV"].append(val_m.mae_eV)
        curves["test_mae_eV"].append(test_m.mae_eV)
        curves["train_r_tot_median"].append(train_m.median_r_tot_pct)
        curves["val_r_tot_median"].append(val_m.median_r_tot_pct)
        curves["test_r_tot_median"].append(test_m.median_r_tot_pct)
        curves["train_abs_err_eV"].append(train_m.mean_abs_err_eV)
        curves["val_abs_err_eV"].append(val_m.mean_abs_err_eV)
        curves["test_abs_err_eV"].append(test_m.mean_abs_err_eV)

        score = val_m.median_r_tot_pct if val_m.n_rtot > 0 else val_m.mae_eV
        if score < best_val_rtot:
            best_val_rtot = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step(val_m.mae_eV)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            dt = time.time() - t0
            print(
                f"[{name}] ep {epoch:04d} | val MAE {val_m.mae_eV:.4f} eV | "
                f"val R_tot med {val_m.median_r_tot_pct:.1f}% | "
                f"test R_tot med {test_m.median_r_tot_pct:.1f}% | {dt:.1f}s",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_train = evaluate_graph_total(model, train_loader, device, target_mean, target_std)
    final_val = evaluate_graph_total(model, val_loader, device, target_mean, target_std)
    final_test = evaluate_graph_total(model, test_loader, device, target_mean, target_std)

    ckpt_path = os.path.join(ROOT, f"cgcnn_{domain}_graph_total_model.pt")
    torch.save(
        {
            "model_state": best_state,
            "architecture": "gated_graph_total",
            "config": config,
            "target_mean": float(target_mean.cpu()),
            "target_std": float(target_std.cpu()),
            "domain": domain,
            "target": "delta_total_eV",
            "best_val_r_tot_median": best_val_rtot,
            "epochs": epochs,
            "seed": seed,
            "split": "grouped 70/15/15",
            "num_parameters": n_params,
        },
        ckpt_path,
    )
    print(f"[{name}] Saved checkpoint -> {ckpt_path}", flush=True)

    # Store per-graph test predictions for plotting
    test_preds: List[Dict] = []
    model.eval()
    with torch.no_grad():
        for data in test_set:
            data = data.to(device)
            pred = float((model(data).view(-1) * target_std + target_mean).cpu().item())
            true = float(data.y.view(-1).cpu().item())
            err = pred - true
            rtot = 100.0 * abs(err) / abs(true) if abs(true) >= MIN_DELTA_EV else float("nan")
            test_preds.append(
                {
                    "true_eV": true,
                    "pred_eV": pred,
                    "err_eV": err,
                    "abs_err_eV": abs(err),
                    "r_tot_pct": rtot,
                }
            )

    return {
        "domain": domain,
        "checkpoint": ckpt_path,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "best_val_r_tot_median": best_val_rtot,
        "final_train_mae_eV": final_train.mae_eV,
        "final_val_mae_eV": final_val.mae_eV,
        "final_test_mae_eV": final_test.mae_eV,
        "final_train_r_tot_median": final_train.median_r_tot_pct,
        "final_val_r_tot_median": final_val.median_r_tot_pct,
        "final_test_r_tot_median": final_test.median_r_tot_pct,
        "final_test_abs_err_median_eV": final_test.median_abs_err_eV,
        "test_predictions": test_preds,
        **curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CGCNN graph regressor on full-cell total residual."
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--domain",
        choices=["point", "planar", "both"],
        default="both",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output-json", default=CURVES_JSON)
    parser.add_argument("--summary-json", default=SUMMARY_JSON)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-prefix", default=PLOT_PREFIX)
    args = parser.parse_args()

    if not args.skip_build:
        build_datasets(force=args.force_build)
    if args.build_only:
        print("Build-only done.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Split: grouped 70/15/15 seed={args.seed}")

    domains = ["point", "planar"] if args.domain == "both" else [args.domain]
    runs: Dict[str, Dict] = {}

    for domain in domains:
        path = DATASETS[domain]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing dataset: {path}")
        dataset = torch.load(path, weights_only=False)
        runs[domain] = _train_one(
            domain=domain,
            dataset=dataset,
            device=device,
            epochs=args.epochs,
            seed=args.seed,
            config=dict(DEFAULT_CONFIG),
        )
        payload = {
            "epochs": args.epochs,
            "seed": args.seed,
            "split": "grouped 70/15/15",
            "target": "delta_total_eV",
            "checkpoint_metric": "val_r_tot_median",
            "runs": runs,
        }
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    summary = {
        "target": "delta_total_eV (full-cell net residual)",
        "model": "GatedGNNGraphRegressor (pooled scalar output)",
        "split": "grouped 70/15/15",
        "epochs": args.epochs,
        "seed": args.seed,
        "by_domain": {
            domain: {
                "final_test_mae_eV": runs[domain]["final_test_mae_eV"],
                "final_test_r_tot_median": runs[domain]["final_test_r_tot_median"],
                "final_test_abs_err_median_eV": runs[domain]["final_test_abs_err_median_eV"],
                "best_val_r_tot_median": runs[domain]["best_val_r_tot_median"],
                "checkpoint": runs[domain]["checkpoint"],
            }
            for domain in domains
        },
    }
    with open(args.summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved curves -> {args.output_json}")
    print(f"Saved summary -> {args.summary_json}")
    for domain in domains:
        r = runs[domain]
        print(
            f"  {domain}: test R_tot med={r['final_test_r_tot_median']:.1f}% "
            f"test MAE={r['final_test_mae_eV']:.4f} eV "
            f"test |err| med={r['final_test_abs_err_median_eV']:.4f} eV"
        )

    if args.plot:
        plot_script = os.path.join(ROOT, "plot_cgcnn_graph_total.py")
        rc = subprocess.run(
            [
                sys.executable,
                plot_script,
                "--input",
                args.output_json,
                "--output-prefix",
                args.plot_prefix,
            ],
            cwd=ROOT,
        ).returncode
        if rc != 0:
            print(f"[warn] Plot script exited with code {rc} (matplotlib required).")


if __name__ == "__main__":
    main()
