"""Build residual-ΔPE graph datasets for point and planar defects.

Writes separate files so absolute-target datasets are not overwritten:

* ``adv_datasets/cycle34_residual_dataset.pt`` — point defects
* ``planar_pyg_dataset_residual_c14c15.pt`` — planar C14/C15 stacks

Example::

    python build_residual_datasets.py
    python build_residual_datasets.py --point-only
    python build_residual_datasets.py --planar-only
"""

from __future__ import annotations

import argparse
import os

from adv_graph_maker import build_adv_datasets
from planar_graph_maker import build_planar_dataset_with_cycles

ROOT = os.path.dirname(os.path.abspath(__file__))

POINT_OUTPUT = os.path.join(ROOT, "adv_datasets", "cycle34_residual_dataset.pt")
POINT_STATS = os.path.join(ROOT, "adv_datasets", "cycle34_residual_stats.json")
PLANAR_OUTPUT = os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15.pt")
PLANAR_STATS = os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15_stats.json")
DEFECT_ATOMS_JSON = os.path.join(ROOT, "laves_defect_atoms_c14c15.json")
INITIAL_DIR = os.path.join(ROOT, "Laves_Planar_Defects", "SIMULATIONS")
RELAXED_DIR = os.path.join(ROOT, "Laves_Screen", "SIMULATIONS")
SIMULATIONS_DIR = os.path.join(ROOT, "SIMULATIONS")


def build_point_residual(force: bool = False) -> None:
    if os.path.isfile(POINT_OUTPUT) and not force:
        print(f"Point residual dataset exists, skipping: {POINT_OUTPUT}")
        return
    print("=== Building point-defect residual dataset ===")
    build_adv_datasets(
        simulations_dir=SIMULATIONS_DIR,
        output_dir=os.path.join(ROOT, "adv_datasets"),
        variants=["cycle34"],
        target_mode="residual",
    )


def build_planar_residual(force: bool = False) -> None:
    if os.path.isfile(PLANAR_OUTPUT) and not force:
        print(f"Planar residual dataset exists, skipping: {PLANAR_OUTPUT}")
        return
    print("=== Building planar residual dataset (C14/C15) ===")
    build_planar_dataset_with_cycles(
        initial_simulations_dir=INITIAL_DIR,
        relaxed_simulations_dir=RELAXED_DIR,
        output_path=PLANAR_OUTPUT,
        stats_path=PLANAR_STATS,
        defect_atoms_json=DEFECT_ATOMS_JSON,
        target_mode="residual",
        require_initial_pe=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build residual-ΔPE datasets for point and planar defects."
    )
    parser.add_argument("--point-only", action="store_true")
    parser.add_argument("--planar-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if output files already exist.",
    )
    args = parser.parse_args()

    build_point = not args.planar_only
    build_planar = not args.point_only

    if build_point:
        build_point_residual(force=args.force)
    if build_planar:
        build_planar_residual(force=args.force)

    print("Done.")


if __name__ == "__main__":
    main()
