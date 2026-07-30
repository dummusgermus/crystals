"""Compare Graph Transformer vs CGCNN on absolute and residual defect datasets."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))

ORDER = (
    "defect",
    "planar_c14c15",
    "defect_residual",
    "planar_residual_c14c15",
)
LABELS = {
    "defect": "point absolute",
    "planar_c14c15": "planar absolute",
    "defect_residual": "point residual",
    "planar_residual_c14c15": "planar residual",
}
COLORS = {
    "defect": "C0",
    "planar_c14c15": "C1",
    "defect_residual": "C2",
    "planar_residual_c14c15": "C3",
}


def _load(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _merge_cgcnn(
    absolute_defect_json: str,
    absolute_planar_json: str,
    residual_json: str,
) -> Dict[str, Dict]:
    merged: Dict[str, Dict] = {}
    for path in (absolute_defect_json, absolute_planar_json, residual_json):
        if not os.path.isfile(path):
            continue
        payload = _load(path)
        merged.update(payload.get("datasets", {}))
    return merged


def _summary_row(name: str, cgcnn: Optional[Dict], transformer: Optional[Dict]) -> str:
    parts = [f"{name:28s}"]
    if cgcnn and "best_test" in cgcnn:
        parts.append(f"CGCNN {cgcnn['best_test']:.6f}")
    elif cgcnn and "test" in cgcnn:
        bi = min(range(len(cgcnn["val"])), key=lambda i: cgcnn["val"][i])
        parts.append(f"CGCNN {cgcnn['test'][bi]:.6f}")
    else:
        parts.append("CGCNN n/a")
    if transformer and "best_test" in transformer:
        parts.append(f"Transformer {transformer['best_test']:.6f}")
    else:
        parts.append("Transformer n/a")
    return " | ".join(parts)


def plot_test_curves(
    cgcnn: Dict[str, Dict],
    transformer: Dict[str, Dict],
    metric: str,
    output: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    keys = [k for k in ORDER if k in cgcnn or k in transformer]
    n = len(keys)
    if n == 0:
        raise SystemExit("No overlapping dataset keys to plot.")

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        if key in cgcnn and cgcnn[key].get("test"):
            series: List[float] = cgcnn[key]["test"]
            ax.plot(
                range(1, len(series) + 1),
                series,
                label="CGCNN",
                color=COLORS[key],
                linewidth=1.8,
            )
        if key in transformer and transformer[key].get("test"):
            series = transformer[key]["test"]
            ax.plot(
                range(1, len(series) + 1),
                series,
                label="Transformer",
                color=COLORS[key],
                linestyle="--",
                linewidth=1.8,
            )
        ax.set_title(LABELS.get(key, key))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Test {metric.upper()}")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("CGCNN vs Graph Transformer (test MAE)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot / print CGCNN vs Transformer benchmark comparison."
    )
    parser.add_argument(
        "--transformer-json",
        type=str,
        default=os.path.join(ROOT, "transformer_absolute_residual_curves.json"),
    )
    parser.add_argument(
        "--absolute-defect-json",
        type=str,
        default=os.path.join(ROOT, "cgcnn_defect_vs_scaled_curves.json"),
    )
    parser.add_argument(
        "--absolute-planar-json",
        type=str,
        default=os.path.join(ROOT, "cgcnn_planar_c14c15_curves.json"),
    )
    parser.add_argument(
        "--residual-json",
        type=str,
        default=os.path.join(ROOT, "cgcnn_residual_both_curves.json"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT, "transformer_vs_cgcnn_curves.png"),
    )
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"), default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.transformer_json):
        raise SystemExit(f"Missing transformer results: {args.transformer_json}")

    tf_payload = _load(args.transformer_json)
    tf_data = tf_payload.get("datasets", {})
    cgcnn_data = _merge_cgcnn(
        args.absolute_defect_json,
        args.absolute_planar_json,
        args.residual_json,
    )
    # Older absolute point JSON may only have curves, not best_*; normalize.
    for key, entry in cgcnn_data.items():
        if "best_test" not in entry and "val" in entry and "test" in entry:
            bi = min(range(len(entry["val"])), key=lambda i: entry["val"][i])
            entry["best_val"] = entry["val"][bi]
            entry["best_test"] = entry["test"][bi]

    metric = str(tf_payload.get("metric", "mae"))
    print("Test MAE @ best validation epoch")
    print("-" * 72)
    for key in ORDER:
        print(_summary_row(key, cgcnn_data.get(key), tf_data.get(key)))

    ylim = tuple(args.ylim) if args.ylim is not None else None
    plot_test_curves(cgcnn_data, tf_data, metric, args.output, ylim=ylim)


if __name__ == "__main__":
    main()
