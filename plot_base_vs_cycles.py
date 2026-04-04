"""Plot test errors over epochs for base vs cycle-augmented datasets."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot test curves from base_vs_cycles_curves.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="base_vs_cycles_curves.json",
        help="JSON file produced by train_base_vs_cycles.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="base_vs_cycles_test_curves.png",
        help="Output PNG filename",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metric = payload.get("metric", "mae").upper()
    curves: Dict[str, Dict[str, List[float]]] = payload["curves"]

    plt.figure(figsize=(8, 5))

    # Consistent order and labels
    order = ["base", "cycle3", "cycle34", "cycle345"]
    labels = {
        "base": "Base",
        "cycle3": "Base + 3-cycles",
        "cycle34": "Base + 3,4-cycles",
        "cycle345": "Base + 3,4,5-cycles",
    }

    for name in order:
        if name not in curves:
            continue
        test = curves[name]["test"]
        epochs = range(1, len(test) + 1)
        plt.plot(epochs, test, label=labels.get(name, name))

    plt.xlabel("Epoch")
    plt.ylabel(f"Test {metric}")
    plt.title(f"Test {metric} over epochs")
    plt.ylim(0.015, 0.022)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()

