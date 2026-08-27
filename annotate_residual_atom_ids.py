"""Annotate existing residual .pt datasets with particle_ids (and point orig_indices).

Does **not** rebuild graphs or change ``x`` / ``y`` / edges — only attaches the
atom-id map needed for clean crystal export. Uses the pure-Python dump reader
from :mod:`crystal_prediction_export` (no OVITO required).

Example::

    python annotate_residual_atom_ids.py
    python annotate_residual_atom_ids.py --point-only
    python annotate_residual_atom_ids.py --force
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

import torch
from torch_geometric.data import Data

from crystal_prediction_export import (
    load_planar_full_cell,
    load_point_full_cell,
    point_config_key,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

POINT_PATH = os.path.join(ROOT, "adv_datasets", "cycle34_residual_dataset.pt")
PLANAR_PATH = os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15.pt")
SIMULATIONS_DIR = os.path.join(ROOT, "SIMULATIONS")
INITIAL_DIR = os.path.join(ROOT, "Laves_Planar_Defects", "SIMULATIONS")
RELAXED_DIR = os.path.join(ROOT, "Laves_Screen", "SIMULATIONS")


def _already_annotated(data: Data) -> bool:
    return getattr(data, "particle_ids", None) is not None


def annotate_point(dataset: List[Data], force: bool) -> List[Data]:
    out: List[Data] = []
    t0 = time.time()
    for i, data in enumerate(dataset):
        if _already_annotated(data) and not force:
            out.append(data)
        else:
            _aids, _pei, _pet, graph_pids, graph_pos = load_point_full_cell(
                data, SIMULATIONS_DIR
            )
            if len(graph_pids) != int(data.num_nodes):
                raise RuntimeError(
                    f"Point id count mismatch {data.folder}/{point_config_key(data)}: "
                    f"{len(graph_pids)} vs {data.num_nodes}"
                )
            data.particle_ids = torch.tensor(graph_pids, dtype=torch.long)
            data.orig_indices = torch.tensor(graph_pos, dtype=torch.long)
            out.append(data)
        if (i + 1) % 50 == 0 or i + 1 == len(dataset):
            print(
                f"  [point] {i + 1}/{len(dataset)} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
    return out


def annotate_planar(dataset: List[Data], force: bool) -> List[Data]:
    out: List[Data] = []
    t0 = time.time()
    for i, data in enumerate(dataset):
        if _already_annotated(data) and not force:
            out.append(data)
        else:
            _aids, _pei, _pet, graph_pids, graph_pos = load_planar_full_cell(
                data, INITIAL_DIR, RELAXED_DIR
            )
            if len(graph_pids) != int(data.num_nodes):
                raise RuntimeError(
                    f"Planar id count mismatch {data.folder}: "
                    f"{len(graph_pids)} vs {data.num_nodes}"
                )
            data.particle_ids = torch.tensor(graph_pids, dtype=torch.long)
            # Full-cell graphs: orig index == node index in dump order.
            data.orig_indices = torch.tensor(graph_pos, dtype=torch.long)
            out.append(data)
        if (i + 1) % 200 == 0 or i + 1 == len(dataset):
            print(
                f"  [planar] {i + 1}/{len(dataset)} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach particle_ids to residual datasets without rebuilding."
    )
    parser.add_argument("--point-only", action="store_true")
    parser.add_argument("--planar-only", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-annotate even if particle_ids already present.",
    )
    args = parser.parse_args()

    do_point = not args.planar_only
    do_planar = not args.point_only

    if do_point:
        if not os.path.isfile(POINT_PATH):
            raise SystemExit(f"Missing point dataset: {POINT_PATH}")
        if not os.path.isdir(SIMULATIONS_DIR):
            raise SystemExit(f"Missing simulations dir: {SIMULATIONS_DIR}")
        print(f"Annotating point dataset: {POINT_PATH}")
        dataset = torch.load(POINT_PATH, weights_only=False)
        dataset = annotate_point(dataset, force=args.force)
        torch.save(dataset, POINT_PATH)
        g0 = dataset[0]
        print(
            f"  Saved {len(dataset)} graphs; sample particle_ids="
            f"{tuple(g0.particle_ids.shape)}"
        )

    if do_planar:
        if not os.path.isfile(PLANAR_PATH):
            raise SystemExit(f"Missing planar dataset: {PLANAR_PATH}")
        if not os.path.isdir(INITIAL_DIR) or not os.path.isdir(RELAXED_DIR):
            raise SystemExit(
                f"Missing planar dirs:\n  {INITIAL_DIR}\n  {RELAXED_DIR}"
            )
        print(f"Annotating planar dataset: {PLANAR_PATH}")
        dataset = torch.load(PLANAR_PATH, weights_only=False)
        dataset = annotate_planar(dataset, force=args.force)
        torch.save(dataset, PLANAR_PATH)
        g0 = dataset[0]
        print(
            f"  Saved {len(dataset)} graphs; sample particle_ids="
            f"{tuple(g0.particle_ids.shape)}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
