"""Plot broad ISF/ESF vs C14/C15 planar residual training curves."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))

ORDER = ("planar_residual_isf", "planar_residual_c14c15")
LABELS = {
    "planar_residual_isf": "broad ISF/ESF sites",
    "planar_residual_c14c15": "C14/C15 deviation sites",
}
COLORS = {
    "planar_residual_isf": "C0",
    "planar_residual_c14c15": "C1",
}
# Color by model; dashed defect def uses a nearby shade of the same hue.
COMBINED_COLORS = {
    ("CGCNN", "planar_residual_isf"): "#1f77b4",
    ("CGCNN", "planar_residual_c14c15"): "#6baed6",
    ("Transformer", "planar_residual_isf"): "#ff7f0e",
    ("Transformer", "planar_residual_c14c15"): "#ffbb78",
}
DEF_STYLES = {
    "planar_residual_isf": "-",
    "planar_residual_c14c15": "--",
}


def _load(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _collect_panels(
    cgcnn_json: Optional[str],
    transformer_json: Optional[str],
) -> Tuple[List[Tuple[str, Dict[str, Dict]]], str]:
    panels: List[Tuple[str, Dict[str, Dict]]] = []
    metric = "MAE"
    if cgcnn_json and os.path.isfile(cgcnn_json):
        payload = _load(cgcnn_json)
        metric = str(payload.get("metric", "mae")).upper()
        panels.append(("CGCNN", payload.get("datasets", {})))
    if transformer_json and os.path.isfile(transformer_json):
        payload = _load(transformer_json)
        metric = str(payload.get("metric", metric)).upper()
        panels.append(("Transformer", payload.get("datasets", {})))
    if not panels:
        raise SystemExit("No curve JSON files found to plot.")
    return panels, metric


def _plot_panel(
    ax,
    datasets: Dict[str, Dict],
    split: str,
    metric: str,
    title: str,
) -> None:
    for key in ORDER:
        if key not in datasets:
            continue
        series: List[float] = datasets[key].get(split, [])
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
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_comparison(
    cgcnn_json: Optional[str],
    transformer_json: Optional[str],
    output: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    panels, metric = _collect_panels(cgcnn_json, transformer_json)

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.5 * len(panels), 4.5), sharey=True, squeeze=False
    )
    for col, (model_name, datasets) in enumerate(panels):
        ax = axes[0][col]
        _plot_panel(
            ax,
            datasets,
            "test",
            metric,
            title=model_name,
        )
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.suptitle("Test MAE", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output}")


def plot_combined(
    cgcnn_json: Optional[str],
    transformer_json: Optional[str],
    output: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """All four curves in one axes: color by model, linestyle by defect def."""
    panels, metric = _collect_panels(cgcnn_json, transformer_json)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for model_name, datasets in panels:
        for key in ORDER:
            if key not in datasets:
                continue
            series: List[float] = datasets[key].get("test", [])
            if not series:
                continue
            ax.plot(
                range(1, len(series) + 1),
                series,
                label=f"{model_name} — {LABELS.get(key, key)}",
                color=COMBINED_COLORS.get((model_name, key)),
                linestyle=DEF_STYLES.get(key, "-"),
                linewidth=1.8,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Test MAE", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ISF/ESF vs C14/C15 planar residual curves."
    )
    parser.add_argument(
        "--cgcnn-json",
        type=str,
        default=os.path.join(
            ROOT, "cgcnn_planar_defect_defs_residual_1000_curves.json"
        ),
    )
    parser.add_argument(
        "--transformer-json",
        type=str,
        default=os.path.join(
            ROOT, "transformer_planar_defect_defs_residual_1000_curves.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT, "planar_defect_defs_residual_1000_curves.png"),
    )
    parser.add_argument(
        "--combined-output",
        type=str,
        default=os.path.join(
            ROOT, "planar_defect_defs_residual_1000_curves_combined.svg"
        ),
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only write the single-panel combined figure.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.005, 0.02),
    )
    args = parser.parse_args()
    ylim = tuple(args.ylim)
    if not args.combined_only:
        plot_comparison(
            args.cgcnn_json,
            args.transformer_json,
            args.output,
            ylim=ylim,
        )
    plot_combined(
        args.cgcnn_json,
        args.transformer_json,
        args.combined_output,
        ylim=ylim,
    )


if __name__ == "__main__":
    main()
