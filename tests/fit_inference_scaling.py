from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit linear scaling for inference timing.")
    parser.add_argument("--input", type=str, default="inference_timing.json")
    parser.add_argument("--output", type=str, default="inference_timing_fit.png")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    per_node = data.get("per_node_count", {})
    if not per_node:
        raise SystemExit("No per_node_count data found in timing JSON.")

    node_counts = sorted(int(k) for k in per_node.keys())
    avg_ms = np.array([per_node[str(n)]["avg_ms"] for n in node_counts], dtype=float)
    x = np.array(node_counts, dtype=float)

    # Fit y = m*x + b
    m, b = np.polyfit(x, avg_ms, 1)
    y_fit = m * x + b

    plt.figure(figsize=(8, 5))
    plt.plot(x, avg_ms, label="Avg ms", marker="o", linewidth=1.0, markersize=3)
    plt.plot(x, y_fit, label=f"Fit: {m:.4f} ms/node + {b:.2f} ms", linewidth=2.0)
    plt.xlabel("Number of nodes in graph")
    plt.ylabel("Inference time (ms)")
    plt.title("Inference time vs node count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Slope: {m:.6f} ms/node, Intercept: {b:.3f} ms")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
