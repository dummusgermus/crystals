"""Plot test MAE curves and per-epoch runtime from tower_sweep_cycle34.json."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import matplotlib.pyplot as plt


COLORS = {
    "k=1 (base)": "C0",
    "k=2": "C1",
    "k=4": "C2",
    "k=8": "C3",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot curves from train_tower_sweep_cycle34.py output JSON."
    )
    parser.add_argument("--input", type=str, default="tower_sweep_cycle34.json")
    parser.add_argument("--output", type=str, default="tower_sweep_cycle34_plot.png")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metric = str(payload.get("metric", "mae")).upper()
    models: Dict[str, Dict[str, List[float]]] = payload["models"]
    ordered_keys = sorted(models.keys(), key=lambda k: int(k.split("=")[1].split()[0]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: test MAE curves
    ax = axes[0]
    for key in ordered_keys:
        series = models[key].get("test", [])
        if not series:
            continue
        xs = range(1, len(series) + 1)
        ax.plot(xs, series, label=key, color=COLORS.get(key), alpha=0.85)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Test {metric}")
    ax.set_title(f"Test {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: per-epoch wall-clock time
    ax = axes[1]
    for key in ordered_keys:
        times = models[key].get("epoch_times", [])
        if not times:
            continue
        xs = range(1, len(times) + 1)
        ax.plot(xs, times, label=key, color=COLORS.get(key), alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch time (s)")
    ax.set_title("Per-epoch wall-clock time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary footer
    lines = []
    for key in ordered_keys:
        m = models[key]
        final_test = m["test"][-1] if m["test"] else float("nan")
        total_t = m.get("total_train_time", 0)
        n_params = m.get("num_parameters", "?")
        lines.append(
            f"{key:15s}:  test {metric} = {final_test:.4f}  |  "
            f"time = {total_t:.0f}s  |  params = {n_params:,}"
        )
    fig.text(
        0.5, 0.01, "\n".join(lines),
        ha="center", va="bottom", fontsize=8, family="monospace",
    )

    fig.suptitle("Tower count sweep on cycle34 (bidirectional)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04 + 0.02 * len(lines), 1, 0.96])
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
