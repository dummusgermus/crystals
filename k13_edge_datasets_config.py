"""Paths for k=13 point datasets with varying internal edge wiring (edge_k).

Safe to import on the cluster (no OVITO).
"""

from __future__ import annotations

import os
from typing import Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
ADV_DIR = os.path.join(ROOT, "adv_datasets")
MANIFEST_JSON = os.path.join(ADV_DIR, "point_k13_edge_manifest.json")
POINT_FULL_CELL_ATOMS = 3000

FIXED_CUTOFF_K = 13

# Production wiring only (edge_k>3 needs intractable cycle re-counting at k=13).
EDGE_K_TIERS: Tuple[Tuple[str, int], ...] = (("e03", 3),)

LOSS_MODES: Tuple[str, ...] = ("graph", "full")


def dataset_path(tag: str) -> str:
    return os.path.join(ADV_DIR, f"cycle34_residual_totals_k13_edge_{tag}.pt")


def stats_path(tag: str) -> str:
    return os.path.join(ADV_DIR, f"cycle34_residual_totals_k13_edge_{tag}_stats.json")
