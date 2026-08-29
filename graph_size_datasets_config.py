"""Paths and tier list for point graph-size datasets (no OVITO imports).

Safe to import on the cluster for training / preflight checks.
"""

from __future__ import annotations

import os
from typing import Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
ADV_DIR = os.path.join(ROOT, "adv_datasets")
MANIFEST_JSON = os.path.join(ADV_DIR, "point_graph_size_manifest.json")
POINT_FULL_CELL_ATOMS = 3000

GRAPH_SIZE_TIERS: Tuple[Tuple[str, int], ...] = (
    ("k08", 8),
    ("k09", 9),
    ("k10", 10),
    ("k11", 11),
    ("k12", 12),
    ("k13", 13),
    ("k22", 22),
)


def dataset_path(tag: str) -> str:
    return os.path.join(ADV_DIR, f"cycle34_residual_totals_size_{tag}.pt")


def stats_path(tag: str) -> str:
    return os.path.join(ADV_DIR, f"cycle34_residual_totals_size_{tag}_stats.json")
