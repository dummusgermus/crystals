"""Point vs planar residual experiment (complete data, 1000 epochs).

1. Force-rebuilds residual-ΔPE datasets from the full simulation archives
   (point: ``SIMULATIONS/``; planar: ``Laves_Planar_Defects`` + ``Laves_Screen``
   with C14/C15 defect labels and ``require_initial_pe=True``).
2. Trains the current best gated CGCNN on both datasets for 1000 epochs.
3. Trains the current best Graph Transformer on both for 1000 epochs.
4. Writes a short comparison summary JSON.

Example::

    python run_point_vs_planar_residual_1000.py
    python run_point_vs_planar_residual_1000.py --skip-build
    python run_point_vs_planar_residual_1000.py --models cgcnn
    python run_point_vs_planar_residual_1000.py --models transformer --epochs 1000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))

POINT_DS = os.path.join(ROOT, "adv_datasets", "cycle34_residual_dataset.pt")
PLANAR_DS = os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15.pt")

CGCNN_JSON = os.path.join(ROOT, "cgcnn_point_vs_planar_residual_1000_curves.json")
TRANSFORMER_JSON = os.path.join(
    ROOT, "transformer_point_vs_planar_residual_1000_curves.json"
)
SUMMARY_JSON = os.path.join(ROOT, "point_vs_planar_residual_1000_summary.json")

DEFAULT_EPOCHS = 1000
DEFAULT_SCORE_LAST_N = 50


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
    return float(var**0.5)


def _run(cmd: List[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def build_datasets(force: bool = True) -> None:
    cmd = [sys.executable, os.path.join(ROOT, "build_residual_datasets.py")]
    if force:
        cmd.append("--force")
    _run(cmd)
    for path, label in ((POINT_DS, "point"), (PLANAR_DS, "planar")):
        if not os.path.isfile(path):
            raise SystemExit(f"Expected {label} residual dataset missing: {path}")
        print(f"[build] {label}: {path}", flush=True)


def train_cgcnn(
    epochs: int,
    seed: int,
    metric: str,
    output_json: str,
) -> None:
    _run(
        [
            sys.executable,
            "-u",
            os.path.join(ROOT, "train_cgcnn_residual_both.py"),
            "--skip-build",
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--metric",
            metric,
            "--output-json",
            output_json,
        ]
    )


def train_transformer(
    epochs: int,
    seed: int,
    metric: str,
    score_last_n: int,
    output_json: str,
    config_json: Optional[str],
    save_checkpoints: bool,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        os.path.join(ROOT, "train_transformer_absolute_residual.py"),
        "--datasets",
        "residual",
        "--architecture",
        "graph",
        "--epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--metric",
        metric,
        "--score-last-n",
        str(score_last_n),
        "--output-json",
        output_json,
    ]
    if config_json:
        cmd.extend(["--config-json", config_json])
    if save_checkpoints:
        cmd.append("--save-checkpoints")
    _run(cmd)


def _enrich_last_n(payload: Dict, score_last_n: int) -> Dict:
    """Add last-N MAE stats if the training script did not already write them."""
    for name, curves in payload.get("datasets", {}).items():
        if "last_n_val_mean" in curves:
            continue
        val = curves.get("val") or []
        test = curves.get("test") or []
        curves["last_n"] = score_last_n
        curves["last_n_val_mean"] = _mean_last_n(val, score_last_n)
        curves["last_n_test_mean"] = _mean_last_n(test, score_last_n)
        curves["last_n_val_std"] = _std_last_n(val, score_last_n)
        curves["last_n_test_std"] = _std_last_n(test, score_last_n)
        payload["datasets"][name] = curves
    return payload


def _load_json(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_summary(
    *,
    cgcnn_json: Optional[str],
    transformer_json: Optional[str],
    score_last_n: int,
    output_json: str,
) -> Dict:
    models: Dict[str, Dict] = {}
    for label, path in (("cgcnn", cgcnn_json), ("transformer", transformer_json)):
        if not path or not os.path.isfile(path):
            continue
        payload = _enrich_last_n(_load_json(path), score_last_n)
        # Persist enriched CGCNN curves so plotting/summary stay consistent.
        if label == "cgcnn":
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        rows = {}
        for name, curves in payload.get("datasets", {}).items():
            rows[name] = {
                "num_graphs": curves.get("num_graphs"),
                "dataset_path": curves.get("dataset_path"),
                "best_val": curves.get("best_val"),
                "best_test": curves.get("best_test"),
                "final_val": curves.get("final_val"),
                "final_test": curves.get("final_test"),
                "last_n_val_mean": curves.get("last_n_val_mean"),
                "last_n_test_mean": curves.get("last_n_test_mean"),
                "last_n_val_std": curves.get("last_n_val_std"),
                "last_n_test_std": curves.get("last_n_test_std"),
            }
        models[label] = {
            "curves_json": path,
            "config": payload.get("config"),
            "epochs": payload.get("epochs"),
            "datasets": rows,
        }

    summary = {
        "score_last_n": score_last_n,
        "datasets": {
            "point_residual": POINT_DS,
            "planar_residual_c14c15": PLANAR_DS,
        },
        "models": models,
        "notes": (
            "Residual ΔPE prediction on complete point and planar archives. "
            "Planar uses C14/C15 defect labels and requires initial per-atom PE. "
            f"Primary score = mean of last {score_last_n} epoch MAEs."
        ),
    }
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Point vs planar residual (last-N MAE) ===", flush=True)
    for model_name, model in models.items():
        print(f"\n[{model_name}]", flush=True)
        for ds_name, row in model["datasets"].items():
            print(
                f"  {ds_name:28s} "
                f"val={row['last_n_val_mean']:.6f}±{row['last_n_val_std']:.6f}  "
                f"test={row['last_n_test_mean']:.6f}±{row['last_n_test_std']:.6f}  "
                f"(n={row.get('num_graphs')})",
                flush=True,
            )
            # How much better is point vs planar (lower MAE = better).
        ds = model["datasets"]
        if "defect_residual" in ds and "planar_residual_c14c15" in ds:
            p = ds["defect_residual"]["last_n_test_mean"]
            pl = ds["planar_residual_c14c15"]["last_n_test_mean"]
            if pl and pl > 0:
                ratio = p / pl
                print(
                    f"  point/planar test MAE ratio: {ratio:.3f} "
                    f"(<1 ⇒ point easier / better MAE)",
                    flush=True,
                )
    print(f"\nSummary written to {output_json}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild complete residual datasets and train best CGCNN + "
            "transformer for 1000 epochs on point and planar."
        )
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric", type=str, default="mae", choices=["mae", "rmse", "mse"]
    )
    parser.add_argument("--score-last-n", type=int, default=DEFAULT_SCORE_LAST_N)
    parser.add_argument(
        "--models",
        type=str,
        default="both",
        choices=["both", "cgcnn", "transformer"],
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--no-force-build",
        action="store_true",
        help="Do not rebuild if residual dataset files already exist.",
    )
    parser.add_argument(
        "--transformer-config-json",
        type=str,
        default=None,
        help=(
            "Optional JSON with best_config / config for the transformer "
            "(e.g. transformer_hparam_sweep.json). Defaults to GRAPH_CONFIG."
        ),
    )
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--cgcnn-json", type=str, default=CGCNN_JSON)
    parser.add_argument("--transformer-json", type=str, default=TRANSFORMER_JSON)
    parser.add_argument("--summary-json", type=str, default=SUMMARY_JSON)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only recompute summary from existing curve JSONs.",
    )
    args = parser.parse_args()

    if args.summary_only:
        write_summary(
            cgcnn_json=args.cgcnn_json,
            transformer_json=args.transformer_json,
            score_last_n=args.score_last_n,
            output_json=args.summary_json,
        )
        return

    if not args.skip_build:
        build_datasets(force=not args.no_force_build)
    else:
        for path, label in ((POINT_DS, "point"), (PLANAR_DS, "planar")):
            if not os.path.isfile(path):
                raise SystemExit(
                    f"Missing {label} residual dataset: {path}. "
                    "Run without --skip-build first."
                )

    do_cgcnn = args.models in ("both", "cgcnn")
    do_tf = args.models in ("both", "transformer")

    if do_cgcnn:
        train_cgcnn(
            epochs=args.epochs,
            seed=args.seed,
            metric=args.metric,
            output_json=args.cgcnn_json,
        )
    if do_tf:
        train_transformer(
            epochs=args.epochs,
            seed=args.seed,
            metric=args.metric,
            score_last_n=args.score_last_n,
            output_json=args.transformer_json,
            config_json=args.transformer_config_json,
            save_checkpoints=args.save_checkpoints,
        )

    write_summary(
        cgcnn_json=args.cgcnn_json if do_cgcnn else None,
        transformer_json=args.transformer_json if do_tf else None,
        score_last_n=args.score_last_n,
        output_json=args.summary_json,
    )


if __name__ == "__main__":
    main()
