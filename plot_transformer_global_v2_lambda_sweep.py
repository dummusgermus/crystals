"""Plot Transformer global v2 lambda sweep curves and summary bars."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_cgcnn_lambda_sweep import METRIC_PANELS


def _lambda_runs(runs: Dict[str, Dict], domain: str) -> List[Tuple[float, str, Dict]]:
    out: List[Tuple[float, str, Dict]] = []
    prefix = f"{domain}_transformer_global_v2_lambda_"
    for key, run in runs.items():
        if not key.startswith(prefix):
            continue
        lam = float(run.get("lambda_tot", key.split("_lambda_", 1)[-1]))
        out.append((lam, key, run))
    out.sort(key=lambda t: t[0])
    return out


def _best_lambda_key(summary: dict, domain: str) -> Optional[str]:
    block = summary.get("best_by_domain", {}).get(domain)
    if not block:
        return None
    return block.get("run_key")


def _plot_epoch_curves(payload: dict, output_prefix: str) -> None:
    runs = payload["runs"]
    summary = payload.get("summary", {})
    domains = ("point", "planar")
    nrows = len(METRIC_PANELS)
    ncols = len(domains)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.0 * nrows), sharex="col")
    if nrows == 1:
        axes = np.array([axes])

    cmap = plt.get_cmap("plasma")

    for col, domain in enumerate(domains):
        lambda_runs = _lambda_runs(runs, domain)
        best_key = _best_lambda_key(summary, domain)
        if not lambda_runs:
            continue
        lam_vals = [t[0] for t in lambda_runs]
        lam_min = min(lam_vals)
        lam_max = max(lam_vals)

        for row, (curve_key, ylabel) in enumerate(METRIC_PANELS):
            ax = axes[row, col]
            for lam, key, run in lambda_runs:
                if curve_key not in run:
                    continue
                color = cmap((lam - lam_min) / (lam_max - lam_min + 1e-9))
                lw = 2.5 if key == best_key else 1.2
                ax.plot(
                    run[curve_key],
                    color=color,
                    linewidth=lw,
                    alpha=0.95 if key == best_key else 0.75,
                    label=f"λ={lam:g}" + (" *best*" if key == best_key else ""),
                )
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(domain)
            if row == nrows - 1:
                ax.set_xlabel("Epoch")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        "Transformer global v2 λ sweep (checkpoint: val R_tot median)",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    out = f"{output_prefix}_epoch_curves.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def _plot_checkpoint_bars(payload: dict, output_prefix: str) -> None:
    runs = payload["runs"]
    summary = payload.get("summary", {})
    domains = ("point", "planar")
    fields = (
        ("final_test_mae", "Test MAE (eV/atom)"),
        ("final_test_r_tot_median", r"Test $R_{\mathrm{tot}}$ median (%)"),
    )

    fig, axes = plt.subplots(len(domains), len(fields), figsize=(12, 4 * len(domains)))
    if len(domains) == 1:
        axes = np.array([axes])

    for row, domain in enumerate(domains):
        lambda_runs = _lambda_runs(runs, domain)
        best_key = _best_lambda_key(summary, domain)
        labels = [f"λ={lam:g}" for lam, _, _ in lambda_runs]
        x = np.arange(len(labels))

        for col, (field, title) in enumerate(fields):
            ax = axes[row, col]
            vals = []
            colors = []
            for lam, key, run in lambda_runs:
                vals.append(float(run.get(field, np.nan)))
                colors.append("C3" if key == best_key else "C0")
            ax.bar(x, vals, color=colors, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_title(f"{domain} — {title}")
            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Checkpoint test metrics (red = best λ by test R_tot median)",
        y=1.01,
    )
    fig.tight_layout()
    out = f"{output_prefix}_best_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="transformer_global_v2_lambda_sweep_curves.json")
    parser.add_argument(
        "--output-prefix",
        default="transformer_global_v2_lambda_sweep",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        payload = json.load(fh)

    _plot_epoch_curves(payload, args.output_prefix)
    _plot_checkpoint_bars(payload, args.output_prefix)


if __name__ == "__main__":
    main()
