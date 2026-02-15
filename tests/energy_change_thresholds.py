import argparse
import os

import numpy as np

from energy_change_histogram import collect_energy_changes


def _percentage_over_threshold(values: np.ndarray, threshold: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sum(values > threshold)) / float(values.size) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute percentages of atoms above energy-change thresholds."
    )
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--simulations-dir",
        default=os.path.join(root_dir, "SIMULATIONS"),
        help="Path to the SIMULATIONS directory.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.07, 0.17],
        help="Energy-change thresholds (eV).",
    )
    parser.add_argument(
        "--signed",
        action="store_true",
        help="Use signed energy changes (default is absolute).",
    )
    args = parser.parse_args()

    use_absolute = not args.signed
    energy_changes, stats = collect_energy_changes(
        simulations_dir=args.simulations_dir,
        use_absolute=use_absolute,
    )

    if energy_changes.size == 0:
        raise ValueError("No energy change values collected; nothing to report.")

    print(
        f"Computed {energy_changes.size} atom energy changes from "
        f"{stats['unrelaxed_pairs_used']} unrelaxed/relaxed pairs."
    )
    if stats["missing_relaxed_atoms"] > 0:
        print(f"Skipped {stats['missing_relaxed_atoms']} atoms missing relaxed PE.")

    label = "absolute" if use_absolute else "signed"
    for threshold in args.thresholds:
        percent = _percentage_over_threshold(energy_changes, threshold)
        count = int(np.sum(energy_changes > threshold))
        print(
            f"Threshold {threshold:.5f} eV ({label}): "
            f"{count} atoms -> {percent:.4f}%"
        )


if __name__ == "__main__":
    main()
