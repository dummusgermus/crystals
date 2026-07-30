"""Plot absolute vs residual CGCNN training curves for point and planar defects."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))

LABELS = {
    "defect": "point defect (absolute)",
    "planar_c14c15": "planar defect (absolute)",
    "defect_residual": "point defect (residual)",
    "planar_residual_c14c15": "planar defect (residual)",
}
COLORS = {
    "defect": "C0",
    "planar_c14c15": "C1",
    "defect_residual": "C0",
    "planar_residual_c14c15": "C1",
}
LINESTYLES = {
    "defect": "-",
    "planar_c14c15": "-",
    "defect_residual": "--",
    "planar_residual_c14c15": "--",
}
ORDER = (
    "defect",
    "planar_c14c15",
    "defect_residual",
    "planar_residual_c14c15",
)

PLOT_TITLE = "CGCNN - absolute vs residual targets (point & planar defects)"


def _load_json(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_all_datasets(
    absolute_defect_json: str,
    absolute_planar_json: str,
    residual_json: str,
) -> Tuple[Dict, str]:
    merged: Dict[str, Dict] = {}
    metric = "mae"

    for path in (absolute_defect_json, absolute_planar_json, residual_json):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        payload = _load_json(path)
        metric = str(payload.get("metric", metric))
        merged.update(payload.get("datasets", {}))

    return merged, metric.upper()


def plot_series(
    ax,
    datasets: Dict[str, Dict],
    curve_name: str,
    metric: str,
    title: str | None = None,
    ylim: Tuple[float, float] | None = None,
) -> None:
    for key in ORDER:
        if key not in datasets:
            continue
        series: List[float] = datasets[key].get(curve_name, [])
        if not series:
            continue
        ax.plot(
            range(1, len(series) + 1),
            series,
            label=LABELS.get(key, key),
            color=COLORS.get(key),
            linestyle=LINESTYLES.get(key, "-"),
            linewidth=1.8,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{curve_name.capitalize()} {metric}")
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot absolute and residual CGCNN curves for point and planar defects."
        )
    )
    parser.add_argument(
        "--absolute-defect-json",
        type=str,
        default=os.path.join(ROOT, "cgcnn_defect_vs_planar_curves.json"),
        help="JSON with point-defect absolute curves (key: defect).",
    )
    parser.add_argument(
        "--absolute-planar-json",
        type=str,
        default=os.path.join(ROOT, "cgcnn_planar_c14c15_curves.json"),
        help="JSON with planar absolute curves (key: planar_c14c15).",
    )
    parser.add_argument(
        "--residual-json",
        type=str,
        default=os.path.join(ROOT, "cgcnn_residual_both_curves.json"),
        help="JSON with residual curves for both defect types.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT, "cgcnn_absolute_vs_residual_curves.png"),
    )
    parser.add_argument(
        "--curve",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional y-axis limits, e.g. --ylim 0 0.05",
    )
    args = parser.parse_args()

    datasets, metric = load_all_datasets(
        args.absolute_defect_json,
        args.absolute_planar_json,
        args.residual_json,
    )
    ylim = tuple(args.ylim) if args.ylim is not None else None

    if args.curve == "all":
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
        for ax, name in zip(axes, ("train", "val", "test")):
            plot_series(
                ax,
                datasets,
                name,
                metric,
                title=f"{name.capitalize()} {metric}",
                ylim=ylim,
            )
        fig.suptitle(PLOT_TITLE, fontsize=13)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_series(ax, datasets, args.curve, metric, ylim=ylim)
        fig.suptitle(PLOT_TITLE, fontsize=13)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
