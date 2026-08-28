"""Plot CGCNN lambda sweep: all metrics vs baseline and best-lambda summary."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


METRIC_PANELS = (
    ("val_mae", "Val MAE (eV/atom)"),
    ("test_mae", "Test MAE (eV/atom)"),
    ("val_r_tot_median", r"Val $R_{\mathrm{tot}}$ median (%)"),
    ("test_r_tot_median", r"Test $R_{\mathrm{tot}}$ median (%)"),
    ("val_r_tot_mean", r"Val $R_{\mathrm{tot}}$ mean (%)"),
    ("test_r_tot_mean", r"Test $R_{\mathrm{tot}}$ mean (%)"),
    ("val_abs_total_err_eV", r"Val $|\Delta E_{\mathrm{err}}|$ (eV)"),
    ("test_abs_total_err_eV", r"Test $|\Delta E_{\mathrm{err}}|$ (eV)"),
)


def _lambda_runs(runs: Dict[str, Dict], domain: str) -> List[Tuple[float, str, Dict]]:
    out: List[Tuple[float, str, Dict]] = []
    prefix = f"{domain}_lambda_"
    for key, run in runs.items():
        if not key.startswith(prefix):
            continue
        lam = float(run.get("lambda_tot", key.split("_lambda_", 1)[-1]))
        out.append((lam, key, run))
    out.sort(key=lambda t: t[0])
    return out


def _best_lambda_key(summary: dict, domain: str) -> str | None:
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
        baseline_key = f"{domain}_atom_only"
        lambda_runs = _lambda_runs(runs, domain)
        best_key = _best_lambda_key(summary, domain)
        if not lambda_runs and baseline_key not in runs:
            continue
        lam_vals = [t[0] for t in lambda_runs]
        lam_min = min(lam_vals) if lam_vals else 0.0
        lam_max = max(lam_vals) if lam_vals else 1.0

        for row, (curve_key, ylabel) in enumerate(METRIC_PANELS):
            ax = axes[row, col]
            if baseline_key in runs and curve_key in runs[baseline_key]:
                ax.plot(
                    runs[baseline_key][curve_key],
                    color="black",
                    linewidth=2.0,
                    label="baseline (atom-only)",
                )
            for lam, key, run in lambda_runs:
                if curve_key not in run:
                    continue
                color = cmap(
                    (lam - lam_min) / (lam_max - lam_min + 1e-9)
                )
                lw = 2.5 if key == best_key else 1.2
                ls = "-" if key == best_key else "-"
                ax.plot(
                    run[curve_key],
                    color=color,
                    linewidth=lw,
                    linestyle=ls,
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
        "CGCNN lambda sweep — epoch curves (70/15/15 split)",
        y=1.03,
        fontsize=13,
    )
    fig.tight_layout()
    out = f"{output_prefix}_epoch_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _bar_summary(payload: dict, output_prefix: str) -> None:
    runs = payload["runs"]
    summary = payload.get("summary", {})
    domains = ("point", "planar")
    fields = (
        ("best_test_mae", "Test MAE (eV/atom)"),
        ("best_test_r_tot_median", r"Test $R_{\mathrm{tot}}$ median (%)"),
    )

    fig, axes = plt.subplots(len(domains), len(fields), figsize=(12, 4 * len(domains)))
    if len(domains) == 1:
        axes = np.array([axes])

    for row, domain in enumerate(domains):
        baseline_key = f"{domain}_atom_only"
        lambda_runs = _lambda_runs(runs, domain)
        best_key = _best_lambda_key(summary, domain)
        labels = ["baseline"] + [f"λ={lam:g}" for lam, _, _ in lambda_runs]
        x = np.arange(len(labels))

        for col, (field, title) in enumerate(fields):
            ax = axes[row, col]
            vals = []
            colors = []
            if baseline_key in runs:
                vals.append(float(runs[baseline_key].get(field, np.nan)))
                colors.append("black")
            for lam, key, run in lambda_runs:
                vals.append(float(run.get(field, np.nan)))
                colors.append("C3" if key == best_key else "C0")
            ax.bar(x, vals, color=colors, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_title(f"{domain} — {title}")
            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Best-over-epoch test metrics (★ red = best λ by val R_tot median)", y=1.01)
    fig.tight_layout()
    out = f"{output_prefix}_best_vs_baseline_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_best_vs_baseline_summary(payload: dict, output_prefix: str) -> None:
    summary = payload.get("summary", {}).get("baseline_vs_best", {})
    if not summary:
        return

    domains = list(summary.keys())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, field, title in zip(
        axes,
        ("best_test_mae", "best_test_r_tot_median", "best_val_r_tot_median"),
        ("Test MAE", r"Test $R_{\mathrm{tot}}$ med.", r"Val $R_{\mathrm{tot}}$ med."),
    ):
        baseline_vals = []
        best_vals = []
        labels = []
        for domain in domains:
            block = summary[domain]
            labels.append(domain)
            baseline_vals.append(block["baseline"][field])
            best_vals.append(block["best_lambda"][field])
        x = np.arange(len(labels))
        ax.bar(x - 0.18, baseline_vals, width=0.36, label="baseline")
        ax.bar(x + 0.18, best_vals, width=0.36, label="best λ")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Best lambda (by val R_tot median) vs atom-only baseline")
    fig.tight_layout()
    out = f"{output_prefix}_best_lambda_vs_baseline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="cgcnn_lambda_sweep_curves.json")
    parser.add_argument("--output-prefix", default="cgcnn_lambda_sweep")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as fh:
        payload = json.load(fh)

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)
    _plot_epoch_curves(payload, args.output_prefix)
    _bar_summary(payload, args.output_prefix)
    _plot_best_vs_baseline_summary(payload, args.output_prefix)


if __name__ == "__main__":
    main()
