"""Plot train/val/test curves and per-epoch runtime from tower_sweep_cycle34.json."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _model_order(
    models: Dict[str, Any],
    tower_counts: Optional[List[int]] = None,
) -> List[str]:
    """Return model keys sorted by k (extracted from labels like ``k=2``)."""

    def sort_key(name: str) -> Tuple[int, str]:
        if "=" in name:
            try:
                k_str = name.split("=", 1)[1].split()[0].strip("()")
                return (int(k_str), name)
            except ValueError:
                pass
        return (10**9, name)

    keys = list(models.keys())
    if tower_counts is not None:
        # Prefer JSON order when labels match k=… pattern
        ordered: List[str] = []
        for k in tower_counts:
            for key in keys:
                if key.startswith(f"k={k}") and key not in ordered:
                    ordered.append(key)
                    break
        for key in sorted(keys, key=sort_key):
            if key not in ordered:
                ordered.append(key)
        return ordered
    return sorted(keys, key=sort_key)


def plot_tower_sweep_from_json(
    payload: Dict[str, Any],
    output_path: str,
    curve: str = "test",
    dpi: int = 150,
) -> None:
    """Load results dict (as from ``tower_sweep_cycle34.json``) and save a figure.

    Parameters
    ----------
    payload
        Parsed JSON with ``metric``, ``models``, optional ``tower_counts``,
        optional ``final_test_mae``.
    output_path
        Where to write the PNG.
    curve
        ``"test"``, ``"val"``, ``"train"``, or ``"all"`` (four subplots: three MAE + runtime).
    dpi
        Figure DPI.
    """

    metric = str(payload.get("metric", "mae")).upper()
    models: Dict[str, Dict[str, List[float]]] = payload["models"]
    tower_counts = payload.get("tower_counts")
    ordered_keys = _model_order(models, tower_counts=tower_counts)

    prop = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    colors = {k: prop[i % len(prop)] for i, k in enumerate(ordered_keys)}

    final_from_json = payload.get("final_test_mae")

    def plot_mae_series(ax, curve_name: str, title: str | None = None) -> None:
        for key in ordered_keys:
            series = models[key].get(curve_name, [])
            if not series:
                continue
            xs = range(1, len(series) + 1)
            ax.plot(xs, series, label=key, color=colors[key], alpha=0.85, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{curve_name.capitalize()} {metric}")
        if title:
            ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    def plot_epoch_times(ax) -> None:
        for key in ordered_keys:
            times = models[key].get("epoch_times", [])
            if not times:
                continue
            xs = range(1, len(times) + 1)
            ax.plot(xs, times, label=key, color=colors[key], alpha=0.7, linewidth=1.0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Epoch time (s)")
        ax.set_title("Per-epoch wall-clock time")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    def summary_lines() -> List[str]:
        lines: List[str] = []
        for key in ordered_keys:
            m = models[key]
            if final_from_json and key in final_from_json:
                final_test = float(final_from_json[key])
            else:
                final_test = m["test"][-1] if m.get("test") else float("nan")
            total_t = m.get("total_train_time", 0)
            n_params = m.get("num_parameters", "?")
            if isinstance(n_params, int):
                params_s = f"{n_params:,}"
            else:
                params_s = str(n_params)
            lines.append(
                f"{key:18s}  test {metric} = {final_test:.4f}  |  "
                f"time = {total_t:.0f}s  |  params = {params_s}"
            )
        return lines

    lines = summary_lines()
    bottom_frac = 0.05 + 0.022 * max(len(lines), 1)

    if curve == "all":
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        plot_mae_series(axes[0][0], "train", title=f"Train {metric}")
        plot_mae_series(axes[0][1], "val", title=f"Val {metric}")
        plot_mae_series(axes[1][0], "test", title=f"Test {metric}")
        plot_epoch_times(axes[1][1])
        fig.text(
            0.5, 0.01, "\n".join(lines),
            ha="center", va="bottom", fontsize=7, family="monospace",
        )
        fig.suptitle("Tower count sweep on cycle34 (bidirectional)", fontsize=13)
        fig.tight_layout(rect=[0, bottom_frac, 1, 0.96])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plot_mae_series(axes[0], curve, title=f"{curve.capitalize()} {metric}")
        plot_epoch_times(axes[1])
        fig.text(
            0.5, 0.01, "\n".join(lines),
            ha="center", va="bottom", fontsize=8, family="monospace",
        )
        fig.suptitle("Tower count sweep on cycle34 (bidirectional)", fontsize=13)
        fig.tight_layout(rect=[0, bottom_frac, 1, 0.94])

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot curves from train_tower_sweep_cycle34.py output JSON."
    )
    parser.add_argument("--input", type=str, default="tower_sweep_cycle34.json")
    parser.add_argument("--output", type=str, default="tower_sweep_cycle34_plot.png")
    parser.add_argument(
        "--curve",
        type=str,
        default="test",
        choices=["test", "val", "train", "all"],
        help="Which metric series to emphasize (all = train/val/test + epoch time).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    plot_tower_sweep_from_json(
        payload,
        output_path=args.output,
        curve=args.curve,
        dpi=args.dpi,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
