import argparse
import json
import os
from typing import Dict, List, Tuple


def _load_results(path: str) -> Tuple[str, Dict[str, Dict[str, float | List[float]]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    metric = payload.get("metric", "mae")
    results = payload.get("results", {})
    if not isinstance(results, dict) or not results:
        raise ValueError("No results found in JSON.")
    return metric, results


def _sort_names(names: List[str]) -> List[str]:
    baseline = [n for n in names if n == "baseline"]
    rest = sorted(n for n in names if n != "baseline")
    return baseline + rest


def _curve_for_key(
    values: Dict[str, float | List[float]],
    key: str,
) -> List[float] | None:
    if key not in values:
        return None
    curve = values[key]
    if not isinstance(curve, list):
        return None
    return [float(v) for v in curve]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablation results from JSON.")
    parser.add_argument("--input", type=str, default="ablation_results_with_baseline.json")
    parser.add_argument(
        "--baseline-json",
        type=str,
        default="",
        help="Optional JSON file that contains the baseline results.",
    )
    parser.add_argument(
        "--baseline-key",
        type=str,
        default="baseline",
        help="Key to read baseline results from baseline JSON.",
    )
    parser.add_argument(
        "--curve",
        type=str,
        default="test_curve",
        choices=["train_curve", "val_curve", "test_curve"],
        help="Which curve to plot.",
    )
    parser.add_argument("--output", type=str, default="ablation_results.png")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--fig-width", type=float, default=10.0)
    parser.add_argument("--fig-height", type=float, default=5.0)
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    metric, results = _load_results(input_path)
    if args.baseline_json:
        baseline_path = os.path.abspath(args.baseline_json)
        base_metric, base_results = _load_results(baseline_path)
        if base_metric != metric:
            raise SystemExit(
                f"Baseline metric '{base_metric}' does not match '{metric}'."
            )
        if args.baseline_key not in base_results:
            raise SystemExit(
                f"Baseline key '{args.baseline_key}' not found in {baseline_path}."
            )
        results = dict(results)
        results["baseline"] = base_results[args.baseline_key]
    names = _sort_names(list(results.keys()))

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Install it and rerun.") from exc

    plt.figure(figsize=(args.fig_width, args.fig_height))
    plotted = 0
    max_len = 0
    for name in names:
        curve = _curve_for_key(results[name], args.curve)
        if curve is None:
            continue
        max_len = max(max_len, len(curve))
        plt.plot(curve, label=name)
        plotted += 1

    if plotted == 0:
        raise SystemExit(f"No curves found for key: {args.curve}")

    plt.xlabel("Epoch")
    plt.ylabel(f"MAE")
    plot_title = args.title or f"MAE without certain features"
    plt.title(plot_title)
    plt.ylim(0.0, 0.5)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    output_path = os.path.abspath(args.output)
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
