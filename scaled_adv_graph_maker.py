"""Scale-normalised defect subgraph dataset builder.

Same pipeline as :mod:`graph_maker` + :mod:`adv_graph_maker` (cycle34 features),
but after the defect-centred subgraph is extracted in physical coordinates,
node positions are translated and **uniformly scaled** so the axis-aligned
bounding box fits inside ``[0, 1]^3`` (longest edge length = 1).

Spatial node/edge features (``dist_to_defect``, edge ``distance``) are
recomputed from the scaled coordinates.  Non-spatial features (type,
per-atom PE, ``is_defect``, ``same_type``, ``incident_defect``) and graph
topology are unchanged.  Cycle counts are computed after scaling (topology
only, so identical to the unscaled graph).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from adv_graph_maker import CYCLE_LENGTHS, count_cycles_per_node
from graph_maker import (
    DEFECT_CUTOFF_K,
    DEFECT_CUTOFF_RADIUS,
    EDGE_CUTOFF_RADIUS,
    EDGE_K,
    _build_subgraph,
    _parse_defect_filename,
    build_type_to_z_map,
    save_dataset,
    VACANCY_INDEX,
)

CYCLE34_COLS = (0, 1)


def scale_to_unit_box(pos: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Translate to the origin-min corner, then uniform-scale into ``[0, 1]^3``.

    Returns scaled positions, the uniform scale divisor (bbox extent), and the
    pre-scale translation (bbox minimum corner).
    """
    bbox_min = pos.min(axis=0)
    shifted = pos - bbox_min
    extent = float(shifted.max())
    if extent < 1e-12:
        return shifted.astype(np.float32), 1.0, bbox_min.astype(np.float32)
    return (shifted / extent).astype(np.float32), extent, bbox_min.astype(np.float32)


def _euclidean_distances(pos: np.ndarray) -> np.ndarray:
    n = pos.shape[0]
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            dist[i, j] = d
            dist[j, i] = d
    return dist


def apply_unit_box_scaling(
    x: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Scale subgraph positions to a unit box; update spatial features."""
    pos_np = np.asarray(pos, dtype=np.float32)
    scaled_pos, extent, bbox_min = scale_to_unit_box(pos_np)

    defect_idx = int((x[:, 2] > 0.5).nonzero(as_tuple=True)[0][0].item())
    dist_to_defect = np.linalg.norm(scaled_pos - scaled_pos[defect_idx], axis=1)

    new_x = x.clone()
    new_x[:, 3] = torch.tensor(dist_to_defect, dtype=torch.float)

    new_pos = torch.tensor(scaled_pos, dtype=torch.float)
    dist_matrix = _euclidean_distances(scaled_pos)

    new_edge_attr = edge_attr.clone()
    if edge_index.numel() > 0:
        for k in range(edge_index.size(1)):
            i = int(edge_index[0, k])
            j = int(edge_index[1, k])
            new_edge_attr[k, 0] = float(dist_matrix[i, j])

    meta = {
        "bbox_extent": extent,
        "bbox_min": bbox_min.tolist(),
        "pos_min": scaled_pos.min(axis=0).tolist(),
        "pos_max": scaled_pos.max(axis=0).tolist(),
    }
    return new_x, new_pos, new_edge_attr, meta


def build_scaled_pyg_dataset(
    simulations_dir: str,
    cutoff_k: int = DEFECT_CUTOFF_K,
    edge_k: int = EDGE_K,
    cutoff_radius: float = DEFECT_CUTOFF_RADIUS,
    edge_radius: float = EDGE_CUTOFF_RADIUS,
    cutoff_mode: str = "shell",
) -> List[Data]:
    """Build defect subgraphs with unit-box scaled coordinates."""
    dataset: List[Data] = []

    for folder in sorted(os.listdir(simulations_dir)):
        if folder.endswith("_MIN"):
            continue
        folder_path = os.path.join(simulations_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        file_count = len(
            [
                f
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            ]
        )
        if file_count != 52:
            continue

        try:
            type_to_z = build_type_to_z_map(folder)
        except ValueError as err:
            print(f"  [skip] {folder}: {err}")
            continue

        data_files = []
        for filename in os.listdir(folder_path):
            if not filename.endswith(".data"):
                continue
            parsed = _parse_defect_filename(filename)
            if parsed is None:
                continue
            data_files.append((filename, parsed))

        if not data_files:
            continue

        for filename, parsed in data_files:
            if parsed["relax_state"] != "unrelaxed":
                continue

            base_name = filename[:-5]
            data_path = os.path.join(folder_path, filename)
            dump_path = os.path.join(folder_path, f"{base_name}.dump")
            if not os.path.exists(dump_path):
                continue

            base_key = (
                f"{parsed['defect_id']}-{parsed['from_type']}-"
                f"{parsed['to_type']}-{parsed['wyckoff']}"
            )
            relaxed_dump_path = os.path.join(folder_path, f"relaxed_{base_key}.dump")
            if not os.path.exists(relaxed_dump_path):
                continue

            x, pos, edge_index, edge_attr, y_node, _sub_index, meta = _build_subgraph(
                dump_path,
                relaxed_dump_path=relaxed_dump_path,
                data_path=data_path,
                defect_id=parsed["defect_id"],
                cutoff_k=cutoff_k,
                edge_k=edge_k,
                cutoff_radius=cutoff_radius,
                edge_radius=edge_radius,
                cutoff_mode=cutoff_mode,
            )

            x, pos, edge_attr, scale_meta = apply_unit_box_scaling(
                x, pos, edge_index, edge_attr
            )
            meta.update(scale_meta)

            lammps_types = x[:, 0].long().tolist()
            z_list = [
                VACANCY_INDEX if t == -1 else type_to_z.get(int(t), VACANCY_INDEX)
                for t in lammps_types
            ]

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                y=y_node,
                z=torch.tensor(z_list, dtype=torch.long),
            )
            data.relax_state = parsed["relax_state"]
            data.defect_id = parsed["defect_id"]
            data.from_type = parsed["from_type"]
            data.to_type = parsed["to_type"]
            data.wyckoff = parsed["wyckoff"]
            data.cutoff_k = cutoff_k
            data.edge_k = edge_k
            data.cutoff_radius = cutoff_radius
            data.edge_radius = edge_radius
            data.cutoff_mode = cutoff_mode
            data.folder = folder
            data.meta = meta
            data.scaled_to_unit_box = True
            dataset.append(data)

    return dataset


def build_scaled_adv_dataset(
    simulations_dir: str,
    output_dir: str,
    output_name: str = "scaled_cycle34_dataset.pt",
    stats_name: str = "scaled_cycle34_stats.json",
    cutoff_k: int = DEFECT_CUTOFF_K,
    edge_k: int = EDGE_K,
    cutoff_radius: float = DEFECT_CUTOFF_RADIUS,
    edge_radius: float = EDGE_CUTOFF_RADIUS,
    cutoff_mode: str = "shell",
) -> str:
    """Build unit-box scaled graphs with normalised 3/4-cycle features."""
    print("Building unit-box scaled base dataset …")
    t0 = time.time()
    dataset = build_scaled_pyg_dataset(
        simulations_dir=simulations_dir,
        cutoff_k=cutoff_k,
        edge_k=edge_k,
        cutoff_radius=cutoff_radius,
        edge_radius=edge_radius,
        cutoff_mode=cutoff_mode,
    )
    if not dataset:
        raise RuntimeError("No graphs were built – check the simulations directory.")
    print(f"  {len(dataset)} graphs built in {time.time() - t0:.1f}s")

    extents = [d.meta.get("bbox_extent", 0.0) for d in dataset]
    print(
        f"  Bbox extent before scaling: "
        f"min={min(extents):.3f} max={max(extents):.3f} "
        f"mean={float(np.mean(extents)):.3f} A"
    )

    print("Computing per-node 3/4-cycle counts …")
    t0 = time.time()
    all_cycle_feats: List[np.ndarray] = []
    for i, data in enumerate(dataset):
        feats = count_cycles_per_node(data.edge_index, data.num_nodes, max_cycle_len=4)
        all_cycle_feats.append(feats)
        if (i + 1) % 100 == 0 or i + 1 == len(dataset):
            print(f"  [{i + 1}/{len(dataset)}] graphs processed")
    print(f"  Cycle counting finished in {time.time() - t0:.1f}s")

    all_cycles = np.concatenate(all_cycle_feats, axis=0)
    mean_all = all_cycles.mean(axis=0)
    std_all = all_cycles.std(axis=0)
    std_all[std_all < 1e-8] = 1.0

    v_mean = mean_all[list(CYCLE34_COLS)]
    v_std = std_all[list(CYCLE34_COLS)]
    print(f"  Cycle means (3, 4): {v_mean}")
    print(f"  Cycle stds  (3, 4): {v_std}")

    variant_ds: List[Data] = []
    for data, cyc in zip(dataset, all_cycle_feats):
        normed = (cyc[:, list(CYCLE34_COLS)] - v_mean) / v_std
        new_data = data.clone()
        new_data.x = torch.cat(
            [data.x, torch.tensor(normed, dtype=torch.float)], dim=-1
        )
        variant_ds.append(new_data)

    os.makedirs(output_dir, exist_ok=True)
    ds_path = os.path.join(output_dir, output_name)
    save_dataset(variant_ds, ds_path)

    stats = {
        "scaling": "uniform bbox fit to [0, 1]^3 (longest edge = 1)",
        "cycle_lengths": [CYCLE_LENGTHS[c] for c in CYCLE34_COLS],
        "mean": v_mean.tolist(),
        "std": v_std.tolist(),
        "num_graphs": len(variant_ds),
        "base_feature_dim": int(dataset[0].x.size(-1)),
        "total_feature_dim": int(variant_ds[0].x.size(-1)),
        "bbox_extent_before_scale": {
            "min": float(min(extents)),
            "max": float(max(extents)),
            "mean": float(np.mean(extents)),
        },
    }
    stats_path = os.path.join(output_dir, stats_name)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"  Saved {len(variant_ds)} graphs -> {ds_path}")
    print(f"  Saved stats -> {stats_path}")
    return ds_path


if __name__ == "__main__":
    import argparse

    root_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Build cycle34 PyG dataset from defect simulations with subgraph "
            "positions scaled to a unit bounding box."
        )
    )
    parser.add_argument(
        "--simulations-dir",
        type=str,
        default=os.path.join(root_dir, "SIMULATIONS"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(root_dir, "adv_datasets_scaled"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scaled_cycle34_dataset.pt",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="scaled_cycle34_stats.json",
    )
    parser.add_argument("--cutoff-k", type=int, default=DEFECT_CUTOFF_K)
    parser.add_argument("--edge-k", type=int, default=EDGE_K)
    parser.add_argument("--cutoff-radius", type=float, default=DEFECT_CUTOFF_RADIUS)
    parser.add_argument("--edge-radius", type=float, default=EDGE_CUTOFF_RADIUS)
    parser.add_argument("--cutoff-mode", choices=["shell", "radius"], default="shell")
    args = parser.parse_args()

    build_scaled_adv_dataset(
        simulations_dir=args.simulations_dir,
        output_dir=args.output_dir,
        output_name=args.output,
        stats_name=args.stats_output,
        cutoff_k=args.cutoff_k,
        edge_k=args.edge_k,
        cutoff_radius=args.cutoff_radius,
        edge_radius=args.edge_radius,
        cutoff_mode=args.cutoff_mode,
    )
    print("Done.")
