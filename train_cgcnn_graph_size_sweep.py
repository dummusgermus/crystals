"""CGCNN global-v2 sweep over point graph shell size (cutoff_k tiers).

Tracks per-atom MAE, graph/full-cell R_tot, training wall time, and
test-set inference throughput (forward pass scaling).

Example::

    python train_cgcnn_graph_size_sweep.py --skip-build --epochs 300 --plot
    sbatch --export=ALL,SKIP_BUILD=1 run_cgcnn_graph_size_sweep.slurm
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

from graph_size_datasets_config import (
    GRAPH_SIZE_TIERS,
    MANIFEST_JSON,
    dataset_path,
)
from gnn_models import build_gated_model_from_dataset
from train_cgcnn_extensive_comparison import DEFAULT_CONFIG, _train_one
from train_cgcnn_global_v2 import default_total_loss_config
from train_single import ensure_graph_delta_field, grouped_split_indices, set_seed

ROOT = os.path.dirname(os.path.abspath(__file__))

SWEEP_JSON = os.path.join(ROOT, "cgcnn_graph_size_sweep_curves.json")
SWEEP_SUMMARY_JSON = os.path.join(ROOT, "cgcnn_graph_size_sweep_summary.json")
SWEEP_PLOT_PREFIX = os.path.join(ROOT, "cgcnn_graph_size_sweep")

POINT_LAMBDA = 0.01
DEFAULT_EPOCHS = 300


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _run_key(tag: str) -> str:
    return f"point_graph_size_{tag}"


def _load_manifest() -> dict:
    if not os.path.isfile(MANIFEST_JSON):
        raise SystemExit(
            f"Missing {MANIFEST_JSON}. Run build_point_graph_size_datasets.py first."
        )
    with open(MANIFEST_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _tier_dataset_paths() -> List[Tuple[str, int, str]]:
    rows: List[Tuple[str, int, str]] = []
    for tag, cutoff_k in GRAPH_SIZE_TIERS:
        path = dataset_path(tag)
        if not os.path.isfile(path):
            raise SystemExit(f"Missing dataset for tier {tag}: {path}")
        rows.append((tag, cutoff_k, path))
    return rows


def benchmark_inference(
    dataset,
    device: torch.device,
    config: dict,
    seed: int,
    *,
    batch_size: Optional[int] = None,
    warmup_batches: int = 3,
) -> dict:
    """Timed forward passes on the grouped test split (no training)."""
    set_seed(seed)
    ensure_graph_delta_field(dataset)
    _, _, test_idx = grouped_split_indices(dataset, seed)
    test_set = [dataset[i] for i in test_idx]
    bs = int(batch_size or config["batch_size"])
    loader = DataLoader(test_set, batch_size=bs, shuffle=False)

    model = build_gated_model_from_dataset(
        dataset,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        use_batch_norm=config["use_batch_norm"],
        activation=config["activation"],
        bidirectional=config["bidirectional"],
    ).to(device)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            model(batch)
            if i + 1 >= warmup_batches:
                break

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_graphs = 0
    n_nodes = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            model(batch)
            n_graphs += int(batch.num_graphs)
            n_nodes += int(batch.num_nodes)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {
        "inference_batch_size": bs,
        "inference_test_graphs": n_graphs,
        "inference_test_nodes": n_nodes,
        "inference_wall_s": elapsed,
        "inference_ms_per_graph": 1000.0 * elapsed / max(n_graphs, 1),
        "inference_ms_per_node": 1000.0 * elapsed / max(n_nodes, 1),
        "inference_graphs_per_s": n_graphs / max(elapsed, 1e-9),
    }


def _build_summary(runs: Dict[str, Dict]) -> dict:
    rows = []
    for tag, cutoff_k in GRAPH_SIZE_TIERS:
        key = _run_key(tag)
        run = runs.get(key)
        if not run:
            continue
        rows.append(
            {
                "tag": tag,
                "cutoff_k": cutoff_k,
                "run_key": key,
                "subset_size_median": run.get("subset_size_median"),
                "final_test_mae": run["final_test_mae"],
                "final_test_r_tot_median": run["final_test_r_tot_median"],
                "final_test_r_tot_full_median": run.get(
                    "final_test_r_tot_full_median", float("nan")
                ),
                "final_test_abs_total_err_eV": run.get("final_test_abs_total_err_eV"),
                "final_test_abs_total_err_full_eV": run.get(
                    "final_test_abs_total_err_full_eV", float("nan")
                ),
                "train_wall_s": run.get("train_wall_s"),
                "train_s_per_epoch": run.get("train_s_per_epoch"),
                "inference_wall_s": run.get("inference_wall_s"),
                "inference_ms_per_graph": run.get("inference_ms_per_graph"),
                "inference_graphs_per_s": run.get("inference_graphs_per_s"),
            }
        )
    best_graph = min(rows, key=lambda r: r["final_test_r_tot_median"]) if rows else None
    best_full = (
        min(rows, key=lambda r: r["final_test_r_tot_full_median"]) if rows else None
    )
    return {
        "domain": "point",
        "model": "cgcnn",
        "loss": "global_v2",
        "lambda_tot": POINT_LAMBDA,
        "selection": "shell cutoff_k",
        "metrics": {
            "atom": "final_test_mae",
            "global_graph": "final_test_r_tot_median",
            "global_full_cell": "final_test_r_tot_full_median",
        },
        "tiers": rows,
        "best_by_test_r_tot_graph": best_graph,
        "best_by_test_r_tot_full": best_full,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CGCNN shell cutoff_k sweep.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--tier", choices=[t for t, _ in GRAPH_SIZE_TIERS])
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--output-json", default=SWEEP_JSON)
    parser.add_argument("--summary-json", default=SWEEP_SUMMARY_JSON)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-prefix", default=SWEEP_PLOT_PREFIX)
    args = parser.parse_args()

    if not args.skip_build:
        cmd = [sys.executable, os.path.join(ROOT, "build_point_graph_size_datasets.py")]
        if args.force_build:
            cmd.append("--force")
        if args.tier:
            cmd.extend(["--tier", args.tier])
        subprocess.run(cmd, check=True, cwd=ROOT)
    if args.build_only:
        print("Build-only done.")
        return

    manifest = _load_manifest()
    summaries_by_tag = manifest.get("summaries", {})
    tier_rows = _tier_dataset_paths()
    if args.tier:
        tier_rows = [row for row in tier_rows if row[0] == args.tier]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, tiers={[t[0] for t in tier_rows]}")

    payload: dict = {}
    if args.resume and os.path.isfile(args.output_json):
        with open(args.output_json, encoding="utf-8") as fh:
            payload = json.load(fh)
    runs: Dict[str, Dict] = dict(payload.get("runs", {}))

    total_cfg = default_total_loss_config(POINT_LAMBDA)

    for tag, cutoff_k, ds_path in tier_rows:
        run_key = _run_key(tag)
        dataset = torch.load(ds_path, weights_only=False)
        tier_stats = summaries_by_tag.get(tag, {})

        need_train = run_key not in runs
        need_inference = (
            not args.skip_inference
            and (
                need_train
                or runs.get(run_key, {}).get("inference_wall_s") is None
            )
        )

        if not need_train and not need_inference:
            print(f"[skip] {run_key} (complete)")
            continue

        if need_train:
            print(f"\n>>> train {run_key} cutoff_k={cutoff_k}")
            t0 = time.perf_counter()
            runs[run_key] = _train_one(
                domain="point",
                dataset=dataset,
                device=device,
                epochs=args.epochs,
                metric=args.metric,
                seed=args.seed,
                config=DEFAULT_CONFIG,
                use_total_loss=True,
                lambda_tot=POINT_LAMBDA,
                run_key=run_key,
                checkpoint_metric="r_tot",
                total_loss_config=total_cfg,
                legacy_total_loss=False,
                extra_total_eval_modes=("full",),
            )
            train_s = time.perf_counter() - t0
            runs[run_key]["train_wall_s"] = train_s
            runs[run_key]["train_s_per_epoch"] = train_s / max(args.epochs, 1)
            print(
                f"[{run_key}] train {train_s / 60:.1f} min "
                f"({runs[run_key]['train_s_per_epoch']:.2f} s/epoch)",
                flush=True,
            )
        else:
            print(f"[skip train] {run_key}", flush=True)

        if need_inference:
            print(f"[{run_key}] inference benchmark …", flush=True)
            inf = benchmark_inference(
                dataset, device, DEFAULT_CONFIG, args.seed
            )
            runs[run_key].update(inf)
            print(
                f"[{run_key}] inference {inf['inference_wall_s']:.3f} s on "
                f"{inf['inference_test_graphs']} test graphs "
                f"({inf['inference_ms_per_graph']:.2f} ms/graph, "
                f"{inf['inference_graphs_per_s']:.1f} graphs/s)",
                flush=True,
            )

        runs[run_key].update(
            {
                "dataset_path": ds_path,
                "graph_size_tag": tag,
                "cutoff_k": cutoff_k,
                "subset_size_median": tier_stats.get("subset_size_median"),
                "subset_size_mean": tier_stats.get("subset_size_mean"),
            }
        )
        _save_json(
            args.output_json,
            {
                "version": "cgcnn_graph_size_sweep",
                "runs": runs,
                "epochs": args.epochs,
                "seed": args.seed,
                "lambda_tot": POINT_LAMBDA,
            },
        )

    summary = _build_summary(runs)
    _save_json(
        args.output_json,
        {
            "version": "cgcnn_graph_size_sweep",
            "runs": runs,
            "epochs": args.epochs,
            "seed": args.seed,
            "lambda_tot": POINT_LAMBDA,
            "summary": summary,
        },
    )
    _save_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))

    if args.plot:
        subprocess.run(
            [
                sys.executable,
                os.path.join(ROOT, "plot_cgcnn_graph_size_sweep.py"),
                "--input",
                args.output_json,
                "--output-prefix",
                args.plot_prefix,
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
