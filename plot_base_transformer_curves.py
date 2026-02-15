from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot base vs transformer test curves from JSON output."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="base_transformer_test_curves.json",
        help="Path to JSON file produced by train_base_transformer_json.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="base_transformer_test_curves.png",
        help="Output plot path.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    base_curve = payload.get("base_test_curve", [])
    transformer_curve = payload.get("transformer_test_curve", [])
    metric = str(payload.get("metric", "mae")).upper()

    if not base_curve and not transformer_curve:
        raise SystemExit("No curves found in input JSON.")

    plt.figure(figsize=(8, 5))
    if transformer_curve:
        plt.plot(transformer_curve, label=f"Transformer test {metric}")
    if base_curve:
        plt.plot(base_curve, label=f"Base test {metric}")

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"Test {metric} per Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
