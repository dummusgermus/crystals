"""Hyperparameter sweep for the Graph Transformer on point-residual data.

Samples configs near the current baseline, trains each for ``--sweep-epochs``
(default 500), and ranks by the mean of the last ``--score-last-n`` (default 50)
validation MAEs. Then optionally retrains the winner for ``--final-epochs``
(default 1000) on all four absolute/residual point+planar datasets.

Example::

    python train_transformer_hparam_sweep.py --n-configs 144 --sweep-epochs 500
    python train_transformer_hparam_sweep.py --skip-final
    python train_transformer_hparam_sweep.py --final-only \\
        --best-config-json transformer_hparam_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch

from train_transformer_absolute_residual import (
    DATASETS,
    GRAPH_CONFIG,
    ROOT,
    _train_one,
)

SWEEP_JSON = os.path.join(ROOT, "transformer_hparam_sweep.json")
FINAL_JSON = os.path.join(ROOT, "transformer_best_1000_curves.json")
POINT_RESIDUAL = "defect_residual"

# Discrete search space around the current Graph Transformer baseline.
SEARCH_SPACE: Dict[str, List] = {
    "hidden_dim": [64, 96, 128, 192, 256],
    "num_layers": [2, 3, 4, 5, 6],
    "num_heads": [2, 4, 8],
    "dropout": [0.0, 0.05, 0.1, 0.15, 0.2],
    "attention_dropout": [0.0, 0.05, 0.1, 0.2],
    "activation": ["gelu", "silu"],
    "lr": [2e-4, 3e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3],
    "weight_decay": [0.0, 1e-6, 1e-5, 5e-5, 1e-4],
    "batch_size": [4, 8, 16],
    "warmup_iters": [10, 20, 40, 50],
    "min_lr": [0.0, 1e-6, 1e-5],
    "gradient_norm": [0.5, 1.0, 2.0],
    "scheduler": ["cosine_warmup", "plateau"],
}


def _config_key(cfg: Dict) -> Tuple:
    return tuple(sorted((k, cfg[k]) for k in sorted(cfg.keys())))


def _sample_config(rng: random.Random) -> Dict:
    cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
    # Ensure num_heads divides hidden_dim.
    heads = [h for h in SEARCH_SPACE["num_heads"] if cfg["hidden_dim"] % h == 0]
    cfg["num_heads"] = rng.choice(heads)
    cfg.setdefault("plateau_patience", 16)
    return cfg


def build_configs(n_configs: int, seed: int) -> List[Dict]:
    """Baseline first, then unique random samples near it."""
    rng = random.Random(seed)
    baseline = dict(GRAPH_CONFIG)
    baseline.setdefault("scheduler", "cosine_warmup")
    baseline.setdefault("plateau_patience", 16)

    configs = [baseline]
    seen = {_config_key(baseline)}
    # Cap attempts so we always terminate even if the space is exhausted.
    attempts = 0
    max_attempts = n_configs * 50
    while len(configs) < n_configs and attempts < max_attempts:
        attempts += 1
        cfg = _sample_config(rng)
        key = _config_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
    return configs


def _slim_result(result: Dict, keep_curves: bool) -> Dict:
    out = {
        k: v
        for k, v in result.items()
        if k not in {"train", "val", "test"} or keep_curves
    }
    return out


def _load_json(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: str, payload: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def run_sweep(
    *,
    n_configs: int,
    sweep_epochs: int,
    score_last_n: int,
    seed: int,
    metric: str,
    output_json: str,
    keep_curves: bool,
    resume: bool,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = DATASETS[POINT_RESIDUAL]["path"]
    if not os.path.isfile(dataset_path):
        raise SystemExit(f"Missing dataset: {dataset_path}")

    configs = build_configs(n_configs, seed)
    print(f"Device: {device}")
    print(f"Sweep configs: {len(configs)} | epochs={sweep_epochs} | last_n={score_last_n}")
    print(f"Dataset: {dataset_path}")

    dataset = torch.load(dataset_path, weights_only=False)

    payload: Dict = {
        "metric": metric,
        "seed": seed,
        "sweep_epochs": sweep_epochs,
        "score_last_n": score_last_n,
        "n_configs": len(configs),
        "dataset": POINT_RESIDUAL,
        "dataset_path": dataset_path,
        "search_space": SEARCH_SPACE,
        "runs": [],
        "best_config": None,
        "best_score": None,
    }

    start_idx = 0
    if resume and os.path.isfile(output_json):
        prev = _load_json(output_json)
        payload["runs"] = list(prev.get("runs", []))
        start_idx = len(payload["runs"])
        print(f"Resuming sweep from config index {start_idx}")

    for i in range(start_idx, len(configs)):
        cfg = configs[i]
        name = f"sweep_{i:03d}"
        print(f"\n=== [{i + 1}/{len(configs)}] {name} ===")
        print(f"Config: {cfg}")
        t0 = time.time()
        result = _train_one(
            name=name,
            dataset=dataset,
            device=device,
            epochs=sweep_epochs,
            metric=metric,
            seed=seed,
            architecture="graph",
            config=cfg,
            target_mode="residual",
            checkpoint_path=None,
            score_last_n=score_last_n,
            quiet_every=50,
        )
        dt = time.time() - t0
        score = float(result["last_n_val_mean"])
        entry = {
            "index": i,
            "name": name,
            "config": cfg,
            "score_last_n_val_mean": score,
            "score_last_n_test_mean": float(result["last_n_test_mean"]),
            "score_last_n_val_std": float(result["last_n_val_std"]),
            "score_last_n_test_std": float(result["last_n_test_std"]),
            "final_val": float(result["final_val"]),
            "final_test": float(result["final_test"]),
            "num_params": int(result["num_params"]),
            "wall_time_s": dt,
            "result": _slim_result(result, keep_curves=keep_curves),
        }
        payload["runs"].append(entry)

        ranked = sorted(payload["runs"], key=lambda r: r["score_last_n_val_mean"])
        best = ranked[0]
        payload["best_config"] = best["config"]
        payload["best_score"] = best["score_last_n_val_mean"]
        payload["best_run_index"] = best["index"]
        payload["ranking"] = [
            {
                "index": r["index"],
                "score_last_n_val_mean": r["score_last_n_val_mean"],
                "score_last_n_test_mean": r["score_last_n_test_mean"],
                "score_last_n_val_std": r["score_last_n_val_std"],
            }
            for r in ranked
        ]
        _save_json(output_json, payload)
        print(
            f"[{name}] last-{score_last_n} val={score:.6f} "
            f"test={entry['score_last_n_test_mean']:.6f} "
            f"({dt / 60:.1f} min) | best so far idx={best['index']} "
            f"val={best['score_last_n_val_mean']:.6f}"
        )

    ranked = sorted(payload["runs"], key=lambda r: r["score_last_n_val_mean"])
    print("\n=== Sweep ranking (top 10 by last-N val MAE) ===")
    for r in ranked[:10]:
        print(
            f"  #{r['index']:03d}  val={r['score_last_n_val_mean']:.6f} "
            f"±{r['score_last_n_val_std']:.6f}  "
            f"test={r['score_last_n_test_mean']:.6f}  cfg={r['config']}"
        )
    return payload


def run_final(
    *,
    best_config: Dict,
    final_epochs: int,
    score_last_n: int,
    seed: int,
    metric: str,
    output_json: str,
    save_checkpoints: bool,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Final {final_epochs}-epoch training with best config ===")
    print(f"Device: {device}")
    print(f"Best config: {best_config}")

    results: Dict[str, Dict] = {}
    for name, meta in DATASETS.items():
        path = meta["path"]
        if not os.path.isfile(path):
            raise SystemExit(f"Missing dataset for {name}: {path}")
        print(f"\n=== Final train {name} ===")
        dataset = torch.load(path, weights_only=False)
        ckpt = None
        if save_checkpoints:
            ckpt = os.path.join(ROOT, f"transformer_best_{name}_model.pt")
        curves = _train_one(
            name=name,
            dataset=dataset,
            device=device,
            epochs=final_epochs,
            metric=metric,
            seed=seed,
            architecture="graph",
            config=best_config,
            target_mode=meta["target_mode"],
            checkpoint_path=ckpt,
            score_last_n=score_last_n,
            quiet_every=20,
        )
        curves["dataset_path"] = path
        curves["num_graphs"] = len(dataset)
        results[name] = curves

        payload = {
            "metric": metric,
            "epochs": final_epochs,
            "seed": seed,
            "score_last_n": score_last_n,
            "model": "GraphTransformer",
            "architecture": "graph",
            "best_config": best_config,
            "datasets": {
                k: {
                    kk: vv
                    for kk, vv in v.items()
                }
                for k, v in results.items()
            },
            "notes": (
                "Final transformer run after hparam sweep on point residual. "
                f"Primary score = mean of last {score_last_n} epoch MAEs."
            ),
        }
        _save_json(output_json, payload)
        print(
            f"[{name}] last-{score_last_n} "
            f"val={curves['last_n_val_mean']:.6f}±{curves['last_n_val_std']:.6f}  "
            f"test={curves['last_n_test_mean']:.6f}±{curves['last_n_test_std']:.6f}"
        )

    print(f"\nFinal curves saved to {output_json}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transformer hparam sweep (point residual) + final 1000-epoch run."
    )
    parser.add_argument("--n-configs", type=int, default=144)
    parser.add_argument("--sweep-epochs", type=int, default=500)
    parser.add_argument("--final-epochs", type=int, default=1000)
    parser.add_argument("--score-last-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="mae", choices=["mae", "rmse", "mse"])
    parser.add_argument("--sweep-json", type=str, default=SWEEP_JSON)
    parser.add_argument("--final-json", type=str, default=FINAL_JSON)
    parser.add_argument("--best-config-json", type=str, default=None)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--keep-curves",
        action="store_true",
        help="Store full train/val/test curves for every sweep run (large JSON).",
    )
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    if args.final_only and args.skip_final:
        raise SystemExit("Choose at most one of --final-only / --skip-final")

    best_config: Optional[Dict] = None
    if not args.final_only:
        sweep_payload = run_sweep(
            n_configs=args.n_configs,
            sweep_epochs=args.sweep_epochs,
            score_last_n=args.score_last_n,
            seed=args.seed,
            metric=args.metric,
            output_json=args.sweep_json,
            keep_curves=args.keep_curves,
            resume=args.resume,
        )
        best_config = deepcopy(sweep_payload["best_config"])
        print(f"\nBest config written to {args.sweep_json}")
    else:
        src = args.best_config_json or args.sweep_json
        if not os.path.isfile(src):
            raise SystemExit(f"--final-only needs a config JSON at {src}")
        loaded = _load_json(src)
        if "best_config" in loaded:
            best_config = loaded["best_config"]
        elif "config" in loaded:
            best_config = loaded["config"]
        else:
            best_config = loaded
        print(f"Loaded best config from {src}")

    if args.skip_final:
        return

    assert best_config is not None
    run_final(
        best_config=best_config,
        final_epochs=args.final_epochs,
        score_last_n=args.score_last_n,
        seed=args.seed,
        metric=args.metric,
        output_json=args.final_json,
        save_checkpoints=args.save_checkpoints,
    )


if __name__ == "__main__":
    main()
