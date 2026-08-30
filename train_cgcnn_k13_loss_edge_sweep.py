"""CGCNN comparison: graph vs full-cell global loss on k=13 (edge_k=3).

Compares global-loss v2 (``target_mode=graph``) against full-cell net-energy
loss (``target_mode=full``) at cutoff_k=13 with production edge_k=3 wiring.

Example::

    python train_cgcnn_k13_loss_edge_sweep.py --skip-build --epochs 300 --plot
    sbatch --export=ALL,SKIP_BUILD=1 run_cgcnn_k13_loss_edge_sweep.slurm
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

from k13_edge_datasets_config import (
    EDGE_K_TIERS,
    FIXED_CUTOFF_K,
    LOSS_MODES,
    MANIFEST_JSON,
    dataset_path,
)
from train_cgcnn_extensive_comparison import DEFAULT_CONFIG, _train_one
from train_cgcnn_global_v2 import default_total_loss_config
from train_single import TotalLossConfig

ROOT = os.path.dirname(os.path.abspath(__file__))

SWEEP_JSON = os.path.join(ROOT, "cgcnn_k13_loss_edge_sweep_curves.json")
SWEEP_SUMMARY_JSON = os.path.join(ROOT, "cgcnn_k13_loss_edge_sweep_summary.json")
SWEEP_PLOT_PREFIX = os.path.join(ROOT, "cgcnn_k13_loss_edge_sweep")

POINT_LAMBDA = 0.01
DEFAULT_EPOCHS = 300


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _run_key(edge_tag: str, loss_mode: str) -> str:
    return f"point_k13_{edge_tag}_{loss_mode}"


def _total_loss_config(loss_mode: str, lambda_tot: float) -> TotalLossConfig:
    cfg = default_total_loss_config(lambda_tot)
    if loss_mode not in LOSS_MODES:
        raise ValueError(f"loss_mode must be one of {LOSS_MODES}, got {loss_mode!r}")
    cfg.target_mode = loss_mode
    return cfg


def _load_manifest() -> dict:
    if not os.path.isfile(MANIFEST_JSON):
        raise SystemExit(
            f"Missing {MANIFEST_JSON}. Run build_point_k13_edge_datasets.py first."
        )
    with open(MANIFEST_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _tier_dataset_paths() -> List[Tuple[str, int, str]]:
    rows: List[Tuple[str, int, str]] = []
    for tag, edge_k in EDGE_K_TIERS:
        path = dataset_path(tag)
        if not os.path.isfile(path):
            raise SystemExit(f"Missing dataset for tier {tag}: {path}")
        rows.append((tag, edge_k, path))
    return rows


def _metric_for_mode(run: dict, loss_mode: str, *, full_cell: bool) -> float:
    if full_cell:
        if loss_mode == "full":
            return float(run.get("final_test_r_tot_median", float("nan")))
        return float(run.get("final_test_r_tot_full_median", float("nan")))
    if loss_mode == "graph":
        return float(run.get("final_test_r_tot_median", float("nan")))
    return float(run.get("final_test_r_tot_graph_median", float("nan")))


def _build_summary(runs: Dict[str, Dict]) -> dict:
    rows = []
    for edge_tag, edge_k in EDGE_K_TIERS:
        for loss_mode in LOSS_MODES:
            key = _run_key(edge_tag, loss_mode)
            run = runs.get(key)
            if not run:
                continue
            rows.append(
                {
                    "edge_tag": edge_tag,
                    "edge_k": edge_k,
                    "loss_mode": loss_mode,
                    "run_key": key,
                    "final_test_mae": run["final_test_mae"],
                    "final_test_r_tot_graph_median": _metric_for_mode(
                        run, loss_mode, full_cell=False
                    ),
                    "final_test_r_tot_full_median": _metric_for_mode(
                        run, loss_mode, full_cell=True
                    ),
                    "final_test_abs_total_err_eV": run.get("final_test_abs_total_err_eV"),
                    "final_test_abs_total_err_full_eV": run.get(
                        "final_test_abs_total_err_full_eV", float("nan")
                    ),
                    "train_wall_s": run.get("train_wall_s"),
                    "train_s_per_epoch": run.get("train_s_per_epoch"),
                }
            )

    def _best(filter_mode: Optional[str], *, full_cell: bool) -> Optional[dict]:
        pool = rows if filter_mode is None else [r for r in rows if r["loss_mode"] == filter_mode]
        if not pool:
            return None
        key = "final_test_r_tot_full_median" if full_cell else "final_test_r_tot_graph_median"
        return min(pool, key=lambda r: r[key])

    return {
        "domain": "point",
        "model": "cgcnn",
        "cutoff_k": FIXED_CUTOFF_K,
        "lambda_tot": POINT_LAMBDA,
        "loss_modes": list(LOSS_MODES),
        "edge_k_tiers": [{"tag": t, "edge_k": k} for t, k in EDGE_K_TIERS],
        "metrics": {
            "atom": "final_test_mae",
            "global_graph": "final_test_r_tot_graph_median",
            "global_full_cell": "final_test_r_tot_full_median",
        },
        "runs": rows,
        "best_full_cell_overall": _best(None, full_cell=True),
        "best_full_cell_graph_loss": _best("graph", full_cell=True),
        "best_full_cell_full_loss": _best("full", full_cell=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CGCNN k=13 graph vs full-cell loss × edge_k sweep."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--edge-tier", choices=[t for t, _ in EDGE_K_TIERS])
    parser.add_argument(
        "--loss-mode",
        choices=list(LOSS_MODES),
        help="Train one loss mode only (default: both).",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-json", default=SWEEP_JSON)
    parser.add_argument("--summary-json", default=SWEEP_SUMMARY_JSON)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-prefix", default=SWEEP_PLOT_PREFIX)
    args = parser.parse_args()

    if not args.skip_build:
        cmd = [sys.executable, os.path.join(ROOT, "build_point_k13_edge_datasets.py")]
        if args.force_build:
            cmd.append("--force")
        if args.edge_tier:
            cmd.extend(["--tier", args.edge_tier])
        subprocess.run(cmd, check=True, cwd=ROOT)
    if args.build_only:
        print("Build-only done.")
        return

    manifest = _load_manifest()
    summaries_by_tag = manifest.get("summaries", {})
    tier_rows = _tier_dataset_paths()
    if args.edge_tier:
        tier_rows = [row for row in tier_rows if row[0] == args.edge_tier]

    loss_modes = list(LOSS_MODES)
    if args.loss_mode:
        loss_modes = [args.loss_mode]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device}, edge_tiers={[t[0] for t in tier_rows]}, "
        f"loss_modes={loss_modes}",
        flush=True,
    )

    payload: dict = {}
    if args.resume and os.path.isfile(args.output_json):
        with open(args.output_json, encoding="utf-8") as fh:
            payload = json.load(fh)
    runs: Dict[str, Dict] = dict(payload.get("runs", {}))

    for edge_tag, edge_k, ds_path in tier_rows:
        dataset = torch.load(ds_path, weights_only=False)
        tier_stats = summaries_by_tag.get(edge_tag, {})

        for loss_mode in loss_modes:
            run_key = _run_key(edge_tag, loss_mode)
            if run_key in runs:
                print(f"[skip] {run_key} (already in output json)")
                continue

            total_cfg = _total_loss_config(loss_mode, POINT_LAMBDA)
            extra_modes = ("full",) if loss_mode == "graph" else ("graph",)

            print(
                f"\n>>> train {run_key} cutoff_k={FIXED_CUTOFF_K} "
                f"edge_k={edge_k} target_mode={loss_mode}",
                flush=True,
            )
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
                extra_total_eval_modes=extra_modes,
            )
            train_s = time.perf_counter() - t0
            runs[run_key]["train_wall_s"] = train_s
            runs[run_key]["train_s_per_epoch"] = train_s / max(args.epochs, 1)
            runs[run_key].update(
                {
                    "dataset_path": ds_path,
                    "edge_tag": edge_tag,
                    "edge_k": edge_k,
                    "cutoff_k": FIXED_CUTOFF_K,
                    "loss_mode": loss_mode,
                    "target_mode": loss_mode,
                    "edge_count_median": tier_stats.get("edge_count_median"),
                    "subset_size_median": tier_stats.get("subset_size_median"),
                }
            )
            run = runs[run_key]
            r_graph = run.get(
                "final_test_r_tot_graph_median", run["final_test_r_tot_median"]
            )
            r_full = run.get(
                "final_test_r_tot_full_median", run["final_test_r_tot_median"]
            )
            print(
                f"[{run_key}] train {train_s / 60:.1f} min | "
                f"test MAE {run['final_test_mae']:.4f} | "
                f"R_tot graph {r_graph:.1f}% | R_tot full {r_full:.1f}%",
                flush=True,
            )

            _save_json(
                args.output_json,
                {
                    "version": "cgcnn_k13_loss_edge_sweep",
                    "runs": runs,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "lambda_tot": POINT_LAMBDA,
                    "cutoff_k": FIXED_CUTOFF_K,
                },
            )

    summary = _build_summary(runs)
    _save_json(
        args.output_json,
        {
            "version": "cgcnn_k13_loss_edge_sweep",
            "runs": runs,
            "epochs": args.epochs,
            "seed": args.seed,
            "lambda_tot": POINT_LAMBDA,
            "cutoff_k": FIXED_CUTOFF_K,
            "summary": summary,
        },
    )
    _save_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2))

    if args.plot:
        subprocess.run(
            [
                sys.executable,
                os.path.join(ROOT, "plot_cgcnn_k13_loss_edge_sweep.py"),
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
