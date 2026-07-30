"""Plot train/val/test curves from cgcnn_defect_vs_scaled_curves.json."""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import matplotlib.pyplot as plt


LABELS = {
    "defect": "point defect data",
    "scaled": "scaled point defect data",
}
COLORS = {
    "defect": "C0",
    "scaled": "C1",
}
ORDER = ("defect", "scaled")

PLOT_TITLE = "CGCNN - point defect vs. scaled point defect data"


def plot_series(
    ax,
    datasets: Dict[str, Dict],
    curve_name: str,
    metric: str,
    title: str | None = None,
) -> None:
    for key in ORDER:
        if key not in datasets:
            continue
        entry = datasets[key]
        series: List[float] = entry.get(curve_name, [])
        if not series:
            continue
        ax.plot(
            range(1, len(series) + 1),
            series,
            label=LABELS.get(key, key),
            color=COLORS.get(key),
            linewidth=1.8,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{curve_name.capitalize()} {metric}")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.01, 0.02)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CGCNN full vs scaled defect training curves from JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="cgcnn_defect_vs_scaled_curves.json",
        help="JSON from train_cgcnn_defect_vs_planar.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cgcnn_defect_vs_scaled_curves.png",
        help="Output PNG filename",
    )
    parser.add_argument(
        "--curve",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to plot (all = train/val/test subplots).",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    metric = str(payload.get("metric", "mae")).upper()
    datasets: Dict[str, Dict] = payload["datasets"]

    if args.curve == "all":
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True)
        for ax, name in zip(axes, ("train", "val", "test")):
            plot_series(ax, datasets, name, metric, title=f"{name.capitalize()} {metric}")
        fig.suptitle(PLOT_TITLE, fontsize=13)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_series(ax, datasets, args.curve, metric)
        fig.suptitle(PLOT_TITLE, fontsize=13)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
