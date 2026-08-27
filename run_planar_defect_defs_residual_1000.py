"""Compare planar defect-site definitions on residual ΔPE (complete data).

Builds two identical planar residual datasets that differ only in defect-site
labels (broad ISF/ESF vs C14/C15), trains the current best CGCNN and Graph
Transformer for 1000 epochs on each, writes curves, and plots the comparison.

Example::

    python run_planar_defect_defs_residual_1000.py
    python run_planar_defect_defs_residual_1000.py --skip-build
    python run_planar_defect_defs_residual_1000.py --models cgcnn --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

import torch

from build_planar_residual_label_defs import DEFS, build_all
from train_cgcnn_residual_both import DEFAULT_CONFIG as CGCNN_CONFIG
from train_cgcnn_residual_both import _train_one as train_cgcnn_one
from train_transformer_absolute_residual import GRAPH_CONFIG, _train_one as train_tf_one

ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_EPOCHS = 1000
DEFAULT_SCORE_LAST_N = 50

CGCNN_JSON = os.path.join(ROOT, "cgcnn_planar_defect_defs_residual_1000_curves.json")
TRANSFORMER_JSON = os.path.join(
    ROOT, "transformer_planar_defect_defs_residual_1000_curves.json"
)
SUMMARY_JSON = os.path.join(ROOT, "planar_defect_defs_residual_1000_summary.json")
PLOT_PNG = os.path.join(ROOT, "planar_defect_defs_residual_1000_curves.png")


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


def _enrich_last_n(payload: Dict, score_last_n: int) -> Dict:
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


def _load_tf_config(config_json: Optional[str]) -> Dict:
    config = dict(GRAPH_CONFIG)
    config.setdefault("scheduler", "cosine_warmup")
    if not config_json:
        return config
    with open(config_json, encoding="utf-8") as fh:
        loaded = json.load(fh)
    if "best_config" in loaded:
        config.update(loaded["best_config"])
    elif "config" in loaded:
        config.update(loaded["config"])
    else:
        config.update(loaded)
    return config


def train_cgcnn(
    epochs: int,
    seed: int,
    metric: str,
    score_last_n: int,
    output_json: str,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cgcnn] device={device} epochs={epochs}")
    results: Dict[str, Dict] = {}
    for name, meta in DEFS.items():
        path = meta["dataset"]
        if not os.path.isfile(path):
            raise SystemExit(f"Missing dataset for {name}: {path}")
        print(f"\n=== CGCNN {name} ({meta['label']}) ===")
        dataset = torch.load(path, weights_only=False)
        curves = train_cgcnn_one(
            name=name,
            dataset=dataset,
            device=device,
            epochs=epochs,
            metric=metric,
            seed=seed,
            config=CGCNN_CONFIG,
        )
        curves["dataset_path"] = path
        curves["num_graphs"] = len(dataset)
        curves["defect_definition"] = meta["label"]
        curves["defect_atoms_json"] = meta["json"]
        results[name] = curves

        payload = {
            "metric": metric,
            "epochs": epochs,
            "seed": seed,
            "score_last_n": score_last_n,
            "model": "CGCNN",
            "config": CGCNN_CONFIG,
            "datasets": results,
            "notes": (
                "Planar residual ΔPE; identical graphs; only defect-site "
                "definition differs (broad ISF/ESF vs C14/C15)."
            ),
        }
        payload = _enrich_last_n(payload, score_last_n)
        with open(output_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Updated {output_json}")
    return payload


def train_transformer(
    epochs: int,
    seed: int,
    metric: str,
    score_last_n: int,
    output_json: str,
    config_json: Optional[str],
    save_checkpoints: bool,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _load_tf_config(config_json)
    print(f"[transformer] device={device} epochs={epochs}")
    print(f"[transformer] config={config}")
    results: Dict[str, Dict] = {}
    for name, meta in DEFS.items():
        path = meta["dataset"]
        if not os.path.isfile(path):
            raise SystemExit(f"Missing dataset for {name}: {path}")
        print(f"\n=== Transformer {name} ({meta['label']}) ===")
        dataset = torch.load(path, weights_only=False)
        ckpt = None
        if save_checkpoints:
            ckpt = os.path.join(ROOT, f"transformer_graph_{name}_model.pt")
        curves = train_tf_one(
            name=name,
            dataset=dataset,
            device=device,
            epochs=epochs,
            metric=metric,
            seed=seed,
            architecture="graph",
            config=config,
            target_mode="residual",
            checkpoint_path=ckpt,
            score_last_n=score_last_n,
        )
        curves["dataset_path"] = path
        curves["num_graphs"] = len(dataset)
        curves["defect_definition"] = meta["label"]
        curves["defect_atoms_json"] = meta["json"]
        results[name] = curves

        payload = {
            "metric": metric,
            "epochs": epochs,
            "seed": seed,
            "score_last_n": score_last_n,
            "model": "GraphTransformer",
            "architecture": "graph",
            "config": config,
            "datasets": results,
            "notes": (
                "Planar residual ΔPE; identical graphs; only defect-site "
                "definition differs (broad ISF/ESF vs C14/C15)."
            ),
        }
        with open(output_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Updated {output_json}")
    return payload


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
        with open(path, encoding="utf-8") as fh:
            payload = _enrich_last_n(json.load(fh), score_last_n)
        if label == "cgcnn":
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        rows = {}
        for name, curves in payload.get("datasets", {}).items():
            rows[name] = {
                "label": curves.get("defect_definition", DEFS.get(name, {}).get("label")),
                "num_graphs": curves.get("num_graphs"),
                "best_val": curves.get("best_val"),
                "best_test": curves.get("best_test"),
                "last_n_val_mean": curves.get("last_n_val_mean"),
                "last_n_test_mean": curves.get("last_n_test_mean"),
                "last_n_val_std": curves.get("last_n_val_std"),
                "last_n_test_std": curves.get("last_n_test_std"),
            }
        models[label] = {"curves_json": path, "datasets": rows}

    summary = {
        "score_last_n": score_last_n,
        "definitions": {
            name: {
                "label": meta["label"],
                "json": meta["json"],
                "dataset": meta["dataset"],
            }
            for name, meta in DEFS.items()
        },
        "models": models,
        "notes": (
            "Same planar residual graphs; defect-site definition is the only "
            "difference. Lower last-N MAE is better."
        ),
    }
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Defect-site definition comparison (last-N MAE) ===")
    for model_name, model in models.items():
        print(f"\n[{model_name}]")
        rows = model["datasets"]
        for name, row in rows.items():
            print(
                f"  {name:28s} "
                f"val={row['last_n_val_mean']:.6f}±{row['last_n_val_std']:.6f}  "
                f"test={row['last_n_test_mean']:.6f}±{row['last_n_test_std']:.6f}"
            )
        if "planar_residual_c14c15" in rows and "planar_residual_isf" in rows:
            better = rows["planar_residual_c14c15"]["last_n_test_mean"]
            broad = rows["planar_residual_isf"]["last_n_test_mean"]
            if broad and broad > 0:
                print(
                    f"  C14C15/ISF test MAE ratio: {better / broad:.3f} "
                    f"(<1 ⇒ C14/C15 better)"
                )
    print(f"\nSummary -> {output_json}")
    return summary


def make_plot(
    cgcnn_json: Optional[str],
    transformer_json: Optional[str],
    output_png: str,
) -> None:
    cmd = [
        sys.executable,
        os.path.join(ROOT, "plot_planar_defect_defs.py"),
        "--output",
        output_png,
    ]
    if cgcnn_json and os.path.isfile(cgcnn_json):
        cmd.extend(["--cgcnn-json", cgcnn_json])
    if transformer_json and os.path.isfile(transformer_json):
        cmd.extend(["--transformer-json", transformer_json])
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train CGCNN + transformer on planar residual data with two "
            "defect-site definitions (ISF/ESF vs C14/C15)."
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
        "--force-build",
        action="store_true",
        help="Rebuild base + both relabeled datasets even if files exist.",
    )
    parser.add_argument(
        "--no-force-build",
        action="store_true",
        help="Reuse existing base/relabeled .pt files when present.",
    )
    parser.add_argument("--transformer-config-json", type=str, default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--cgcnn-json", type=str, default=CGCNN_JSON)
    parser.add_argument("--transformer-json", type=str, default=TRANSFORMER_JSON)
    parser.add_argument("--summary-json", type=str, default=SUMMARY_JSON)
    parser.add_argument("--plot-output", type=str, default=PLOT_PNG)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only recompute summary (+ plot) from existing curve JSONs.",
    )
    args = parser.parse_args()

    if args.summary_only:
        write_summary(
            cgcnn_json=args.cgcnn_json,
            transformer_json=args.transformer_json,
            score_last_n=args.score_last_n,
            output_json=args.summary_json,
        )
        if not args.no_plot:
            make_plot(args.cgcnn_json, args.transformer_json, args.plot_output)
        return

    if not args.skip_build:
        # Default: force rebuild so the complete-data experiment is reproducible.
        force = True
        if args.no_force_build:
            force = False
        if args.force_build:
            force = True
        build_all(force=force)
    else:
        for name, meta in DEFS.items():
            if not os.path.isfile(meta["dataset"]):
                raise SystemExit(
                    f"Missing {name} dataset: {meta['dataset']}. "
                    "Run without --skip-build first."
                )

    do_cgcnn = args.models in ("both", "cgcnn")
    do_tf = args.models in ("both", "transformer")

    if do_cgcnn:
        train_cgcnn(
            epochs=args.epochs,
            seed=args.seed,
            metric=args.metric,
            score_last_n=args.score_last_n,
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
    if not args.no_plot:
        make_plot(
            args.cgcnn_json if do_cgcnn else None,
            args.transformer_json if do_tf else None,
            args.plot_output,
        )


if __name__ == "__main__":
    main()
