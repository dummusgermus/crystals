#!/usr/bin/env python3
"""Compare total PE prediction error to the true net system energy change.

For each delivered crystal CSV under predictions/point/ and predictions/planar/,
reads header totals and reports

    |pe_error_*_total| / |pe_true_total - pe_initial_total|

as a percentage. A baseline that predicts no relaxation (pe_pred = pe_initial)
has 100% on this metric by construction.

Usage:
    python analyze_total_error_vs_delta.py
    python analyze_total_error_vs_delta.py --predictions-root predictions --min-delta 1e-6
"""

from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


HEADER_RE = re.compile(r"^#\s*([^=]+)=\s*([+-]?[\d.eE+-]+)\s*$")


@dataclass
class CrystalStats:
    domain: str
    split: str
    config: str
    path: str
    n_atoms: int
    delta_true_eV: float
    pe_error_cgcnn_eV: float
    pe_error_transformer_eV: float
    mae_cgcnn_eV: float
    mae_transformer_eV: float

    @property
    def rel_cgcnn_pct(self) -> float:
        return 100.0 * abs(self.pe_error_cgcnn_eV) / abs(self.delta_true_eV)

    @property
    def rel_transformer_pct(self) -> float:
        return 100.0 * abs(self.pe_error_transformer_eV) / abs(self.delta_true_eV)


def parse_header(path: Path) -> Dict[str, float | str]:
    meta: Dict[str, float | str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("atom_id,"):
                break
            if line.startswith("# domain="):
                meta["domain"] = line.split("=", 1)[1].strip()
            elif line.startswith("# config="):
                meta["config"] = line.split("=", 1)[1].strip()
            elif line.startswith("# split="):
                meta["split"] = line.split("=", 1)[1].strip()
            else:
                m = HEADER_RE.match(line.strip())
                if m:
                    meta[m.group(1).strip()] = float(m.group(2))
    return meta


def load_crystal(path: Path) -> Optional[CrystalStats]:
    meta = parse_header(path)
    required = (
        "pe_initial_total_eV",
        "pe_true_total_eV",
        "pe_error_cgcnn_total_eV",
        "pe_error_transformer_total_eV",
        "mae_cgcnn_abs_eV",
        "mae_transformer_abs_eV",
    )
    if not all(k in meta for k in required):
        return None

    delta = float(meta["pe_true_total_eV"]) - float(meta["pe_initial_total_eV"])
    n_atoms = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("atom_id") or not line.strip():
                continue
            n_atoms += 1

    return CrystalStats(
        domain=str(meta.get("domain", path.parts[-4] if len(path.parts) >= 4 else "")),
        split=str(meta.get("split", "unknown")),
        config=str(meta.get("config", path.stem)),
        path=str(path),
        n_atoms=n_atoms,
        delta_true_eV=delta,
        pe_error_cgcnn_eV=float(meta["pe_error_cgcnn_total_eV"]),
        pe_error_transformer_eV=float(meta["pe_error_transformer_total_eV"]),
        mae_cgcnn_eV=float(meta["mae_cgcnn_abs_eV"]),
        mae_transformer_eV=float(meta["mae_transformer_abs_eV"]),
    )


def iter_csvs(root: Path) -> Iterable[Path]:
    for domain in ("point", "planar"):
        domain_root = root / domain
        if not domain_root.is_dir():
            continue
        yield from sorted(domain_root.rglob("*.csv"))


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    values_sorted = sorted(values)
    n = len(values_sorted)

    def pct(p: float) -> float:
        if n == 1:
            return values_sorted[0]
        idx = (n - 1) * p / 100.0
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return values_sorted[lo] * (1 - frac) + values_sorted[hi] * frac

    return {
        "n": float(n),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": pct(10),
        "p90": pct(90),
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "frac_below_100pct": 100.0 * sum(v < 100.0 for v in values) / n,
        "frac_below_50pct": 100.0 * sum(v < 50.0 for v in values) / n,
    }


def print_block(title: str, cgcnn: List[float], transformer: List[float]) -> None:
    sc = summarize(cgcnn)
    st = summarize(transformer)
    print(f"\n{title}")
    print("-" * len(title))
    if not sc:
        print("  (no crystals)")
        return
    print(f"  n = {int(sc['n'])}")
    for label, key in (
        ("mean", "mean"),
        ("median", "median"),
        ("p10", "p10"),
        ("p90", "p90"),
        ("min", "min"),
        ("max", "max"),
    ):
        print(
            f"  {label:>8}: CGCNN {sc[key]:7.2f}%   Transformer {st[key]:7.2f}%"
        )
    print(
        f"  {'<100%':>8}: CGCNN {sc['frac_below_100pct']:6.1f}% of crystals   "
        f"Transformer {st['frac_below_100pct']:6.1f}% of crystals"
    )
    print(
        f"  {'<50%':>8}: CGCNN {sc['frac_below_50pct']:6.1f}% of crystals   "
        f"Transformer {st['frac_below_50pct']:6.1f}% of crystals"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Total system PE error as % of true net energy change."
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("predictions"),
        help="Root folder containing point/ and planar/ deliveries",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-6,
        help="Skip crystals with |delta_true| below this (eV)",
    )
    args = parser.parse_args()

    crystals: List[CrystalStats] = []
    skipped_tiny_delta = 0
    for csv_path in iter_csvs(args.predictions_root):
        row = load_crystal(csv_path)
        if row is None:
            continue
        if abs(row.delta_true_eV) < args.min_delta:
            skipped_tiny_delta += 1
            continue
        crystals.append(row)

    print("Total system error vs true net energy change")
    print("=" * 48)
    print(f"Predictions root: {args.predictions_root.resolve()}")
    print(f"Metric: 100 * |pe_error_total| / |pe_true_total - pe_initial_total|")
    print("Baseline (predict pe_initial, zero delta): 100% exactly")
    print(f"Skipped (|delta| < {args.min_delta:g} eV): {skipped_tiny_delta}")

    all_c = [c.rel_cgcnn_pct for c in crystals]
    all_t = [c.rel_transformer_pct for c in crystals]
    print_block("All crystals (point + planar)", all_c, all_t)

    for domain in ("point", "planar"):
        subset = [c for c in crystals if c.domain == domain]
        print_block(
            f"{domain} — all",
            [c.rel_cgcnn_pct for c in subset],
            [c.rel_transformer_pct for c in subset],
        )
        for split in ("train", "validation"):
            ss = [c for c in subset if c.split == split]
            print_block(
                f"{domain} — {split}",
                [c.rel_cgcnn_pct for c in ss],
                [c.rel_transformer_pct for c in ss],
            )

    # Worst / best examples for sanity
    if crystals:
        worst_c = max(crystals, key=lambda c: c.rel_cgcnn_pct)
        best_c = min(crystals, key=lambda c: c.rel_cgcnn_pct)
        print("\nExamples (CGCNN relative error)")
        print(f"  best:  {best_c.rel_cgcnn_pct:.2f}%  {best_c.domain}/{best_c.config}")
        print(f"  worst: {worst_c.rel_cgcnn_pct:.2f}%  {worst_c.domain}/{worst_c.config}")


if __name__ == "__main__":
    main()
