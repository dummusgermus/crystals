"""Build identical planar residual datasets for two defect-site definitions.

Geometry, edges, cycles, and residual targets (ΔPE) are shared. Only
``is_defect``, ``dist_to_defect``, and edge ``incident_defect`` differ.

Definitions
-----------
* **broad ISF/ESF** — ``laves_defect_atoms.json``
* **C14/C15 deviation** — ``laves_defect_atoms_c14c15.json`` (current best)

Pipeline
--------
1. Build one complete residual planar dataset from
   ``Laves_Planar_Defects`` (initial PE) + ``Laves_Screen`` (relaxed PE).
2. Relabel that base twice so the two outputs are graph-identical except
   for defect-site features.

Example::

    python build_planar_residual_label_defs.py
    python build_planar_residual_label_defs.py --force
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch

from planar_graph_maker import build_planar_dataset_with_cycles
from relabel_planar_defects import load_defect_atoms_by_stack, relabel_dataset

ROOT = os.path.dirname(os.path.abspath(__file__))

INITIAL_DIR = os.path.join(ROOT, "Laves_Planar_Defects", "SIMULATIONS")
RELAXED_DIR = os.path.join(ROOT, "Laves_Screen", "SIMULATIONS")

BASE_OUTPUT = os.path.join(ROOT, "planar_pyg_dataset_residual_complete_base.pt")
BASE_STATS = os.path.join(ROOT, "planar_pyg_dataset_residual_complete_base_stats.json")

DEFS: Dict[str, Dict[str, str]] = {
    "planar_residual_isf": {
        "json": os.path.join(ROOT, "laves_defect_atoms.json"),
        "dataset": os.path.join(ROOT, "planar_pyg_dataset_residual_isf.pt"),
        "stats": os.path.join(ROOT, "planar_pyg_dataset_residual_isf_stats.json"),
        "label": "broad ISF/ESF defect sites",
    },
    "planar_residual_c14c15": {
        "json": os.path.join(ROOT, "laves_defect_atoms_c14c15.json"),
        "dataset": os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15.pt"),
        "stats": os.path.join(
            ROOT, "planar_pyg_dataset_residual_c14c15_relabel_stats.json"
        ),
        "label": "C14/C15 deviation defect sites",
    },
}


def _verify_identical_except_labels(a, b) -> None:
    """Sanity-check that two datasets differ only in defect-site features."""
    if len(a) != len(b):
        raise RuntimeError(f"Graph count mismatch: {len(a)} vs {len(b)}")
    for i, (ga, gb) in enumerate(zip(a, b)):
        if ga.num_nodes != gb.num_nodes:
            raise RuntimeError(f"Graph {i}: num_nodes mismatch")
        if not torch.equal(ga.y, gb.y):
            raise RuntimeError(f"Graph {i}: residual targets differ")
        if not torch.equal(ga.edge_index, gb.edge_index):
            raise RuntimeError(f"Graph {i}: edges differ")
        # x cols: 0=type, 1=pe, 2=is_defect, 3=dist, 4+=cycles
        if not torch.allclose(ga.x[:, :2], gb.x[:, :2]):
            raise RuntimeError(f"Graph {i}: type/PE features differ")
        if ga.x.size(1) > 4 and not torch.allclose(ga.x[:, 4:], gb.x[:, 4:]):
            raise RuntimeError(f"Graph {i}: cycle features differ")
        # edge_attr cols: 0=dist, 1=same_type, 2=incident_defect
        if ga.edge_attr.numel() and not torch.allclose(
            ga.edge_attr[:, :2], gb.edge_attr[:, :2]
        ):
            raise RuntimeError(f"Graph {i}: non-defect edge attrs differ")


def build_base(force: bool = False):
    if os.path.isfile(BASE_OUTPUT) and not force:
        print(f"Base residual dataset exists, loading: {BASE_OUTPUT}")
        return torch.load(BASE_OUTPUT, weights_only=False)

    print("=== Building complete planar residual base dataset ===")
    # Labels on the base file are temporary; both comparison sets are relabeled.
    build_planar_dataset_with_cycles(
        initial_simulations_dir=INITIAL_DIR,
        relaxed_simulations_dir=RELAXED_DIR,
        output_path=BASE_OUTPUT,
        stats_path=BASE_STATS,
        defect_atoms_json=DEFS["planar_residual_c14c15"]["json"],
        target_mode="residual",
        require_initial_pe=True,
    )
    return torch.load(BASE_OUTPUT, weights_only=False)


def relabel_one(
    name: str,
    dataset: List,
    force: bool = False,
) -> Tuple[str, Dict]:
    meta = DEFS[name]
    out_path = meta["dataset"]
    if os.path.isfile(out_path) and not force:
        print(f"[{name}] exists, skipping: {out_path}")
        return out_path, {}

    mapping = load_defect_atoms_by_stack(json_path=meta["json"])
    print(
        f"[{name}] Relabeling with {os.path.basename(meta['json'])} "
        f"({sum(1 for v in mapping.values() if v)} non-empty stacks) …"
    )
    sim_dir = INITIAL_DIR if os.path.isdir(INITIAL_DIR) else None
    relabeled, stats = relabel_dataset(dataset, mapping, simulations_dir=sim_dir)
    stats["source_dataset"] = os.path.abspath(BASE_OUTPUT)
    stats["defect_atoms_json"] = os.path.abspath(meta["json"])
    stats["label"] = meta["label"]
    stats["target_mode"] = "residual"

    torch.save(relabeled, out_path)
    with open(meta["stats"], "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(
        f"[{name}] Saved {len(relabeled)} graphs -> {out_path} "
        f"(defect atoms total={stats['total_defect_atoms']})"
    )
    return out_path, stats


def build_all(force: bool = False) -> Dict[str, str]:
    base = build_base(force=force)
    paths: Dict[str, str] = {}
    loaded: Dict[str, List] = {}
    for name in DEFS:
        path, _ = relabel_one(name, base, force=force)
        paths[name] = path
        loaded[name] = torch.load(path, weights_only=False)

    print("\n=== Verifying identical geometry/targets across definitions ===")
    keys = list(DEFS.keys())
    _verify_identical_except_labels(loaded[keys[0]], loaded[keys[1]])
    print(
        f"OK: {len(loaded[keys[0]])} graphs match on y/edges/type/PE/cycles; "
        "only defect-site features differ."
    )
    for name, ds in loaded.items():
        n_def = sum(int(g.meta.get("num_defect_atoms", 0)) for g in ds)
        print(f"  {name}: total defect-atom marks = {n_def}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build identical planar residual datasets for broad ISF/ESF vs "
            "C14/C15 defect-site definitions."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild base and relabeled datasets even if files exist.",
    )
    args = parser.parse_args()
    paths = build_all(force=args.force)
    print("\nDone.")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
