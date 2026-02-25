from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Tuple


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _load_runs(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    runs = payload.get("runs", [])
    if not runs:
        raise SystemExit("No runs found in JSON.")
    return payload.get("meta", {}), runs


def _best_overall(runs: List[Dict[str, Any]], score_key: str) -> Dict[str, Any]:
    return min(runs, key=lambda r: float(r.get(score_key, float("inf"))))


def _top_k(runs: List[Dict[str, Any]], score_key: str, k: int) -> List[Dict[str, Any]]:
    return sorted(runs, key=lambda r: float(r.get(score_key, float("inf"))))[:k]


def _best_per_category(
    runs: List[Dict[str, Any]], score_key: str
) -> Dict[str, Dict[str, Any]]:
    # category -> value -> scores
    buckets: Dict[str, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        cfg = run.get("config", {})
        score = float(run.get(score_key, float("inf")))
        for key, value in cfg.items():
            buckets[key][value].append(score)

    summary: Dict[str, Dict[str, Any]] = {}
    for key, value_map in buckets.items():
        ranked = sorted(
            (
                {
                    "value": value,
                    "mean_score": mean(scores),
                    "num_runs": len(scores),
                }
                for value, scores in value_map.items()
            ),
            key=lambda x: x["mean_score"],
        )
        summary[key] = {
            "best_value": ranked[0]["value"],
            "best_mean_score": ranked[0]["mean_score"],
            "ranked_values": ranked,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze base hyperparameter sweep JSON results."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="base_hparam_sweep_curves.json",
        help="Path to sweep JSON file.",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default="test_at_best_val",
        choices=["test_at_best_val", "final_test", "best_val"],
        help="Metric key to rank runs (lower is better).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top full configurations to print.",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="",
        help="Optional path to write analysis summary JSON.",
    )
    args = parser.parse_args()

    meta, runs = _load_runs(args.input)
    best = _best_overall(runs, args.score_key)
    top = _top_k(runs, args.score_key, args.top_k)
    per_category = _best_per_category(runs, args.score_key)

    metric = meta.get("metric", "metric")
    print(f"Analyzing {len(runs)} runs | rank by '{args.score_key}' | metric={metric}")
    print("\n=== Best Overall Configuration ===")
    print(f"{args.score_key}: {_fmt(best.get(args.score_key))}")
    for k, v in best.get("config", {}).items():
        print(f"- {k}: {_fmt(v)}")

    print(f"\n=== Top {len(top)} Configurations ===")
    for i, run in enumerate(top, start=1):
        cfg = run.get("config", {})
        cfg_short = ", ".join(f"{k}={_fmt(v)}" for k, v in cfg.items())
        print(f"{i:02d}. {args.score_key}={_fmt(run.get(args.score_key))} | {cfg_short}")

    print("\n=== Best Value Per Category (averaged over all other parameters) ===")
    for key, info in per_category.items():
        print(
            f"- {key}: best={_fmt(info['best_value'])} | "
            f"mean_{args.score_key}={_fmt(info['best_mean_score'])}"
        )

    if args.output_summary:
        summary = {
            "meta": meta,
            "input": args.input,
            "score_key": args.score_key,
            "best_overall": best,
            "top_k": top,
            "best_per_category": per_category,
        }
        with open(args.output_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved analysis summary to {args.output_summary}")


if __name__ == "__main__":
    main()
