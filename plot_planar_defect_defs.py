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


def _load(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


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
    ax.set_ylabel(f"{split.capitalize()} {metric}")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_comparison(
    cgcnn_json: Optional[str],
    transformer_json: Optional[str],
    output: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    panels: List[Tuple[str, Dict[str, Dict], str]] = []
    metric = "MAE"
    if cgcnn_json and os.path.isfile(cgcnn_json):
        payload = _load(cgcnn_json)
        metric = str(payload.get("metric", "mae")).upper()
        panels.append(("CGCNN", payload.get("datasets", {}), "CGCNN"))
    if transformer_json and os.path.isfile(transformer_json):
        payload = _load(transformer_json)
        metric = str(payload.get("metric", metric)).upper()
        panels.append(("Transformer", payload.get("datasets", {}), "Transformer"))
    if not panels:
        raise SystemExit("No curve JSON files found to plot.")

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.5 * len(panels), 4.5), sharey=True, squeeze=False
    )
    for col, (_key, datasets, model_name) in enumerate(panels):
        ax = axes[0][col]
        _plot_panel(
            ax,
            datasets,
            "test",
            metric,
            title=f"{model_name} — test {metric}",
        )
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.suptitle(
        "Planar residual ΔPE — defect-site definition comparison\n"
        "(identical graphs; only is_defect / dist_to_defect / incident_defect differ)",
        fontsize=12,
    )
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
        "--ylim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.0, 0.02),
    )
    args = parser.parse_args()
    plot_comparison(
        args.cgcnn_json,
        args.transformer_json,
        args.output,
        ylim=tuple(args.ylim),
    )


if __name__ == "__main__":
    main()
