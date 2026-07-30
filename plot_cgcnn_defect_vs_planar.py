"""Plot train/val/test curves from one or more CGCNN comparison JSON files."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


LABELS = {
    "defect": "point defect data",
    "planar": "planar defect data",
    "planar_labeled": "labeled planar defect data",
    "planar_c14c15": "planar defect data (C14/C15)",
    "planar_matrix": "matrix-aligned (no defect labels)",
    "planar_residual_c14c15": "planar residual ΔPE (C14/C15)",
}
COLORS = {
    "defect": "C0",
    "planar": "C1",
    "planar_labeled": "C2",
    "planar_c14c15": "C3",
    "planar_matrix": "C4",
    "planar_residual_c14c15": "C5",
}
ORDER = (
    "defect",
    "planar",
    "planar_labeled",
    "planar_c14c15",
    "planar_matrix",
    "planar_residual_c14c15",
)

PLOT_TITLE = "CGCNN - point defect vs. planar defect data"


def load_merged_payload(json_paths: List[str]) -> Tuple[Dict, Dict[str, Dict]]:
    """Load and merge dataset curves from multiple JSON files."""
    merged_datasets: Dict[str, Dict] = {}
    base_payload: Dict = {}
    missing: List[str] = []

    for path in json_paths:
        if not os.path.isfile(path):
            missing.append(path)
            continue
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not base_payload:
            base_payload = payload
        merged_datasets.update(payload.get("datasets", {}))

    if missing:
        print("Skipping missing curve files:")
        for path in missing:
            print(f"  - {path}")

    if not base_payload:
        raise ValueError("No JSON payloads loaded.")
    base_payload = dict(base_payload)
    base_payload["datasets"] = merged_datasets
    return base_payload, merged_datasets


def plot_series(
    ax,
    datasets: Dict[str, Dict],
    curve_name: str,
    metric: str,
    title: str | None = None,
    order: Tuple[str, ...] = ORDER,
) -> None:
    for key in order:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CGCNN defect vs planar training curves from JSON."
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=[
            "cgcnn_defect_vs_planar_curves.json",
            "cgcnn_planar_labeled_curves.json",
            "cgcnn_planar_new_defs_curves.json",
        ],
        help=(
            "One or more JSON files with training curves.  Later files add "
            "datasets without overwriting keys from earlier files."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cgcnn_defect_vs_planar_curves.png",
        help="Output PNG filename",
    )
    parser.add_argument(
        "--curve",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to plot (all = train/val/test subplots).",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        default=None,
        choices=list(ORDER),
        help="Plot only these dataset keys (default: all present in JSON).",
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

    payload, datasets = load_merged_payload(args.input)
    if args.only:
        datasets = {k: datasets[k] for k in args.only if k in datasets}
        if not datasets:
            raise SystemExit(
                f"None of the requested keys {args.only} found in the input JSONs."
            )
    metric = str(payload.get("metric", "mae")).upper()
    plot_order = tuple(args.only) if args.only else ORDER

    if args.curve == "all":
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True)
        for ax, name in zip(axes, ("train", "val", "test")):
            plot_series(
                ax, datasets, name, metric,
                title=f"{name.capitalize()} {metric}",
                order=plot_order,
            )
            if args.ylim is not None:
                ax.set_ylim(args.ylim[0], args.ylim[1])
        fig.suptitle(PLOT_TITLE, fontsize=13)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_series(ax, datasets, args.curve, metric, order=plot_order)
        if args.ylim is not None:
            ax.set_ylim(args.ylim[0], args.ylim[1])
        fig.suptitle(PLOT_TITLE, fontsize=13)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
