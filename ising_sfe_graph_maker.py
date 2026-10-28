"""Build PyG graphs from the 20261020 Ising / SFE evaluation dataset.

Reads ``relaxed.out`` (LAMMPS ``write_data``) via OVITO and pairs each
structure with ``total_energy.txt`` as a **graph-level** target.

Node features (planar schema minus initial PE)::

    [type, is_defect, dist_to_defect] + normalised 3/4-cycle counts

Edge features (unchanged)::

    [distance, same_type, incident_defect]

Atom-level defect labels are not available for these polytypes / supercells,
so ``is_defect`` and ``dist_to_defect`` are zero (same as unlabeled planar
stacks). Species come from the material folder name ``A-B`` (AB2 Laves).

Default sources (colleague paths)::

    materials/<A-B>/{c14|c15}/{intrinsic|extrinsic}_fault/ising/4_relax_polytypes/*/
    materials/<A-B>/.../supercell/{1_structure_no_defect|2_structure_with_defect}/

Requires the ``jara-ovito`` conda env (OVITO + torch + torch_geometric).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Import torch / PyG before OVITO (imported lazily below) to avoid SSL DLL clashes
# with the pip-installed OVITO wheel in the jara-ovito env.
warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import numpy as np
import torch
from torch_geometric.data import Data

from adv_graph_maker import count_cycles_per_node
from graph_maker import (
    EDGE_K,
    SHELL_TOL_REL,
    SYMBOL_TO_Z,
    VACANCY_INDEX,
    _get_particle_ids,
    _get_property_array,
    save_dataset,
)
from planar_graph_maker import _cell_and_pbc, _shell_threshold

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.normpath(
    os.path.join(ROOT, "..", "data", "20261020_data", "to_send", "materials")
)
DEFAULT_OUTPUT = os.path.join(
    ROOT, "adv_datasets", "ising_sfe_cycle34_total_energy_dataset.pt"
)
DEFAULT_STATS = os.path.join(
    ROOT, "adv_datasets", "ising_sfe_cycle34_total_energy_stats.json"
)

CYCLE_LENGTHS = (3, 4)
CYCLE_COLS = (0, 1)
PHASES = ("c14", "c15")
FAULTS = ("intrinsic_fault", "extrinsic_fault")
KINDS_DEFAULT = ("ising_polytype", "supercell")

MATERIAL_RE = re.compile(r"^([A-Za-z]{1,2})-([A-Za-z]{1,2})$")


def _normalize_symbol(raw: str) -> str:
    s = raw.strip()
    if not s:
        raise ValueError("empty element symbol")
    return s[0].upper() + s[1:].lower()


def parse_material_folder(name: str) -> Tuple[str, str]:
    """Parse ``ac-ca`` / ``Ag-Be`` into ``(Ac, Ca)`` element symbols."""
    m = MATERIAL_RE.match(name)
    if not m:
        raise ValueError(f"Could not parse A-B material folder: {name!r}")
    a, b = _normalize_symbol(m.group(1)), _normalize_symbol(m.group(2))
    if a not in SYMBOL_TO_Z or b not in SYMBOL_TO_Z:
        raise ValueError(f"Element outside H..Fm vocab in {name!r}: {a}, {b}")
    return a, b


def build_type_to_z_map(elem_a: str, elem_b: str) -> Dict[int, int]:
    return {1: SYMBOL_TO_Z[elem_a], 2: SYMBOL_TO_Z[elem_b]}


def _read_scalar(path: str) -> float:
    text = open(path, encoding="utf-8", errors="ignore").read().strip()
    if not text:
        raise ValueError(f"Empty scalar file: {path}")
    return float(text.split()[0])


def _read_sfe_mjm2(path: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    text = open(path, encoding="utf-8", errors="ignore").read().strip()
    if not text:
        return None
    return float(text.split()[0])


def _is_nonempty(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def discover_records(
    materials_root: str,
    kinds: Sequence[str] = KINDS_DEFAULT,
    materials: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """Walk the to_send tree and collect structure/energy record dicts."""
    kind_set = set(kinds)
    records: List[Dict] = []
    mat_filter = {m.lower() for m in materials} if materials else None

    for mat_name in sorted(os.listdir(materials_root)):
        mat_path = os.path.join(materials_root, mat_name)
        if not os.path.isdir(mat_path):
            continue
        if mat_filter is not None and mat_name.lower() not in mat_filter:
            continue
        try:
            elem_a, elem_b = parse_material_folder(mat_name)
        except ValueError:
            continue

        for phase in PHASES:
            phase_path = os.path.join(mat_path, phase)
            if not os.path.isdir(phase_path):
                continue
            for fault in FAULTS:
                fault_path = os.path.join(phase_path, fault)
                if not os.path.isdir(fault_path):
                    continue

                if "ising_polytype" in kind_set:
                    poly_root = os.path.join(
                        fault_path, "ising", "4_relax_polytypes"
                    )
                    if os.path.isdir(poly_root):
                        for poly in sorted(os.listdir(poly_root)):
                            poly_path = os.path.join(poly_root, poly)
                            if not os.path.isdir(poly_path):
                                continue
                            relaxed = os.path.join(poly_path, "relaxed.out")
                            energy = os.path.join(poly_path, "total_energy.txt")
                            if not (_is_nonempty(relaxed) and _is_nonempty(energy)):
                                continue
                            records.append(
                                {
                                    "kind": "ising_polytype",
                                    "material": mat_name,
                                    "element_a": elem_a,
                                    "element_b": elem_b,
                                    "phase": phase,
                                    "fault": fault,
                                    "polytype": poly,
                                    "stack_tag": poly,
                                    "relaxed_out": relaxed,
                                    "total_energy_txt": energy,
                                    "cohesive_energy_txt": os.path.join(
                                        poly_path, "cohesive_energy.txt"
                                    ),
                                    "sfe_txt": os.path.join(
                                        fault_path, "ising", "5_post", "sfe_ising.txt"
                                    ),
                                    "folder": os.path.relpath(
                                        poly_path, materials_root
                                    ),
                                }
                            )

                if "supercell" in kind_set:
                    for tag, is_faulted in (
                        ("1_structure_no_defect", False),
                        ("2_structure_with_defect", True),
                    ):
                        sc_path = os.path.join(fault_path, "supercell", tag)
                        relaxed = os.path.join(sc_path, "relaxed.out")
                        energy = os.path.join(sc_path, "total_energy.txt")
                        if not (_is_nonempty(relaxed) and _is_nonempty(energy)):
                            continue
                        records.append(
                            {
                                "kind": "supercell",
                                "material": mat_name,
                                "element_a": elem_a,
                                "element_b": elem_b,
                                "phase": phase,
                                "fault": fault,
                                "polytype": tag,
                                "stack_tag": tag,
                                "is_faulted": is_faulted,
                                "relaxed_out": relaxed,
                                "total_energy_txt": energy,
                                "cohesive_energy_txt": os.path.join(
                                    sc_path, "cohesive_energy.txt"
                                ),
                                "sfe_txt": os.path.join(
                                    sc_path, "sfe_supercell.txt"
                                ),
                                "folder": os.path.relpath(sc_path, materials_root),
                            }
                        )

                if "reference" in kind_set:
                    ref_path = os.path.join(
                        fault_path, "ising", "2_reference_structure"
                    )
                    relaxed = os.path.join(ref_path, "relaxed.out")
                    energy = os.path.join(ref_path, "total_energy.txt")
                    if _is_nonempty(relaxed) and _is_nonempty(energy):
                        records.append(
                            {
                                "kind": "reference",
                                "material": mat_name,
                                "element_a": elem_a,
                                "element_b": elem_b,
                                "phase": phase,
                                "fault": fault,
                                "polytype": "reference",
                                "stack_tag": "reference",
                                "relaxed_out": relaxed,
                                "total_energy_txt": energy,
                                "cohesive_energy_txt": os.path.join(
                                    ref_path, "cohesive_energy.txt"
                                ),
                                "sfe_txt": os.path.join(
                                    fault_path, "ising", "5_post", "sfe_ising.txt"
                                ),
                                "folder": os.path.relpath(ref_path, materials_root),
                            }
                        )

    return records


def _build_full_graph_from_relaxed(
    relaxed_out: str,
    edge_k: int,
    edge_radius: float,
    edge_mode: str,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict,
    torch.Tensor,
]:
    """Full-cell graph from a relaxed LAMMPS data file (no per-atom PE)."""
    from ovito.io import import_file

    frame = import_file(relaxed_out).compute()
    particle_ids = _get_particle_ids(frame)
    types = _get_property_array(
        frame.particles, ["Particle Type", "Type", "type"]
    )
    if types is None:
        raise ValueError(f"Missing particle types in {relaxed_out}")
    positions = np.asarray(frame.particles.positions, dtype=float)
    n_atoms = len(positions)

    cell_matrix, pbc = _cell_and_pbc(frame)
    try:
        inv_cell = np.linalg.inv(cell_matrix)
    except np.linalg.LinAlgError:
        inv_cell = np.linalg.pinv(cell_matrix)
    frac_positions = positions @ inv_cell

    def _min_image_delta(i_idx: int, j_idx: int) -> np.ndarray:
        dfrac = frac_positions[j_idx] - frac_positions[i_idx]
        for dim in range(3):
            if pbc[dim]:
                dfrac[dim] -= np.round(dfrac[dim])
        return dfrac @ cell_matrix

    # No atom-level fault IDs in this archive → zeros (planar unlabeled case).
    defect_indices: set = set()
    dist_to_defect = np.zeros(n_atoms, dtype=float)

    node_features = [
        [float(types[i]), 0.0, float(dist_to_defect[i])] for i in range(n_atoms)
    ]
    x = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(positions, dtype=torch.float)

    dist_matrix = np.zeros((n_atoms, n_atoms), dtype=float)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = float(np.linalg.norm(_min_image_delta(i, j)))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    edge_index: List[List[int]] = []
    edge_attr: List[List[float]] = []
    edge_set = set()
    edge_dist = 0.0
    for i in range(n_atoms):
        distances = dist_matrix[i]
        neighbor_dists = [d for d in distances if d > 0.0]
        if not neighbor_dists:
            continue
        sorted_dist = np.sort(np.array(neighbor_dists))
        if edge_mode == "shell":
            edge_dist = _shell_threshold(sorted_dist, edge_k)
        elif edge_mode == "radius":
            edge_dist = float(edge_radius)
        else:
            raise ValueError(f"Unsupported edge_mode: {edge_mode}")

        for j in range(n_atoms):
            if i == j:
                continue
            if distances[j] <= edge_dist:
                key = (min(i, j), max(i, j))
                if key in edge_set:
                    continue
                edge_set.add(key)
                edge_index.append([i, j])
                same_type = 1.0 if types[i] == types[j] else 0.0
                incident_defect = (
                    1.0 if (i in defect_indices or j in defect_indices) else 0.0
                )
                edge_attr.append([float(distances[j]), same_type, incident_defect])

    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.zeros((0, 3), dtype=torch.float)

    meta = {
        "num_atoms": n_atoms,
        "edge_shell_distance": float(edge_dist) if n_atoms > 1 else 0.0,
        "edge_mode": edge_mode,
        "has_unrelaxed_pe": False,
        "has_defect_labels": False,
        "num_defect_atoms": 0,
        "defect_ids": [],
        "target_mode": "graph_total_energy",
    }
    particle_ids_tensor = torch.tensor(
        [int(pid) for pid in particle_ids], dtype=torch.long
    )
    return x, pos, edge_index_tensor, edge_attr_tensor, meta, particle_ids_tensor


def build_ising_sfe_pyg_dataset(
    materials_root: str = DEFAULT_DATA_ROOT,
    edge_k: int = EDGE_K,
    edge_radius: float = 3.0,
    edge_mode: str = "shell",
    kinds: Sequence[str] = KINDS_DEFAULT,
    materials: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[Data]:
    """Build one graph per ``relaxed.out`` / ``total_energy.txt`` pair."""
    if not os.path.isdir(materials_root):
        raise FileNotFoundError(f"materials_root not found: {materials_root}")

    print(f"  materials_root: {os.path.abspath(materials_root)}")
    print(f"  kinds={list(kinds)}, edge_mode={edge_mode}, edge_k={edge_k}")
    records = discover_records(materials_root, kinds=kinds, materials=materials)
    if limit is not None:
        records = records[: max(0, int(limit))]
    print(f"  discovered {len(records)} structure/energy pairs")

    dataset: List[Data] = []
    skipped = 0
    t0 = time.time()
    for i, rec in enumerate(records):
        try:
            total_e = _read_scalar(rec["total_energy_txt"])
            cohesive = None
            if _is_nonempty(rec["cohesive_energy_txt"]):
                cohesive = _read_scalar(rec["cohesive_energy_txt"])
            sfe = _read_sfe_mjm2(rec.get("sfe_txt", ""))

            x, pos, edge_index, edge_attr, meta, particle_ids = (
                _build_full_graph_from_relaxed(
                    relaxed_out=rec["relaxed_out"],
                    edge_k=edge_k,
                    edge_radius=edge_radius,
                    edge_mode=edge_mode,
                )
            )
            type_to_z = build_type_to_z_map(rec["element_a"], rec["element_b"])
            lammps_types = x[:, 0].long().tolist()
            z_list = [
                VACANCY_INDEX if t == -1 else type_to_z.get(int(t), VACANCY_INDEX)
                for t in lammps_types
            ]

            y = torch.tensor([[total_e]], dtype=torch.float)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                y=y,
                z=torch.tensor(z_list, dtype=torch.long),
            )
            data.folder = rec["folder"]
            data.material = rec["material"]
            data.element_a = rec["element_a"]
            data.element_b = rec["element_b"]
            data.phase = rec["phase"]
            data.fault = rec["fault"]
            data.kind = rec["kind"]
            data.polytype = rec["polytype"]
            data.stack_sequence = rec["stack_tag"]
            data.edge_k = edge_k
            data.edge_radius = edge_radius
            data.edge_mode = edge_mode
            data.target_mode = "graph_total_energy"
            data.total_energy_eV = torch.tensor([total_e], dtype=torch.float)
            # Always set optional scalars so PyG Batch collate sees uniform keys.
            data.cohesive_energy_eV = torch.tensor(
                [float(cohesive) if cohesive is not None else float("nan")],
                dtype=torch.float,
            )
            data.sfe_mjm2 = torch.tensor(
                [float(sfe) if sfe is not None else float("nan")],
                dtype=torch.float,
            )
            data.is_faulted = bool(rec.get("is_faulted", False))
            # Keep meta keys identical on every graph (PyG Batch collates dicts).
            meta = dict(meta)
            meta["total_energy_eV"] = total_e
            meta["cohesive_energy_eV"] = (
                float(cohesive) if cohesive is not None else float("nan")
            )
            meta["sfe_mjm2"] = float(sfe) if sfe is not None else float("nan")
            data.meta = meta
            data.particle_ids = particle_ids
            dataset.append(data)
        except (ValueError, OSError, RuntimeError) as err:
            skipped += 1
            if skipped <= 20:
                print(f"  [skip] {rec['folder']}: {err}")
            continue

        if (i + 1) % 200 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t0
            print(
                f"  [{i + 1}/{len(records)}] built={len(dataset)} "
                f"skipped={skipped} ({elapsed:.1f}s)",
                flush=True,
            )

    if skipped:
        print(f"  Skipped {skipped} records with read/build errors.")
    return dataset


def build_ising_sfe_dataset_with_cycles(
    materials_root: str = DEFAULT_DATA_ROOT,
    output_path: str = DEFAULT_OUTPUT,
    stats_path: str = DEFAULT_STATS,
    edge_k: int = EDGE_K,
    edge_radius: float = 3.0,
    edge_mode: str = "shell",
    kinds: Sequence[str] = KINDS_DEFAULT,
    materials: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Data], Dict]:
    """Build graphs, append normalised 3/4-cycle features, save ``.pt`` + stats."""
    print("Building Ising/SFE base dataset …")
    t0 = time.time()
    dataset = build_ising_sfe_pyg_dataset(
        materials_root=materials_root,
        edge_k=edge_k,
        edge_radius=edge_radius,
        edge_mode=edge_mode,
        kinds=kinds,
        materials=materials,
        limit=limit,
    )
    if not dataset:
        raise RuntimeError(
            "No graphs built. Check materials_root and that "
            "relaxed.out / total_energy.txt pairs exist."
        )
    print(f"  {len(dataset)} graphs built in {time.time() - t0:.1f}s")

    print("Computing per-node 3/4-cycle counts …")
    t0 = time.time()
    all_cycle_feats: List[np.ndarray] = []
    for i, graph in enumerate(dataset):
        feats = count_cycles_per_node(
            graph.edge_index, graph.num_nodes, max_cycle_len=4
        )
        all_cycle_feats.append(feats)
        if (i + 1) % 500 == 0 or i + 1 == len(dataset):
            print(f"  [{i + 1}/{len(dataset)}] graphs processed", flush=True)
    print(f"  Cycle counting finished in {time.time() - t0:.1f}s")

    all_cycles = np.concatenate(all_cycle_feats, axis=0)
    mean_all = all_cycles.mean(axis=0)
    std_all = all_cycles.std(axis=0)
    std_all[std_all < 1e-8] = 1.0
    v_mean = mean_all[list(CYCLE_COLS)]
    v_std = std_all[list(CYCLE_COLS)]
    print(f"  Cycle means (3, 4): {v_mean}")
    print(f"  Cycle stds  (3, 4): {v_std}")

    augmented: List[Data] = []
    for graph, cyc in zip(dataset, all_cycle_feats):
        normed = (cyc[:, list(CYCLE_COLS)] - v_mean) / v_std
        new_graph = graph.clone()
        new_graph.x = torch.cat(
            [graph.x, torch.tensor(normed, dtype=torch.float)], dim=-1
        )
        augmented.append(new_graph)

    save_dataset(augmented, output_path)

    y_all = torch.cat([g.y.view(1) for g in augmented], dim=0).view(-1)
    kind_counts = {}
    for g in augmented:
        kind_counts[g.kind] = kind_counts.get(g.kind, 0) + 1
    n_atoms = [int(g.num_nodes) for g in augmented]

    stats = {
        "cycle_lengths": list(CYCLE_LENGTHS),
        "mean": v_mean.tolist(),
        "std": v_std.tolist(),
        "num_graphs": len(augmented),
        "kind_counts": kind_counts,
        "base_feature_dim": int(dataset[0].x.size(-1)),
        "total_feature_dim": int(augmented[0].x.size(-1)),
        "node_features": [
            "type",
            "is_defect",
            "dist_to_defect",
            "cycle3_norm",
            "cycle4_norm",
        ],
        "edge_features": ["distance", "same_type", "incident_defect"],
        "edge_k": edge_k,
        "edge_mode": edge_mode,
        "target_mode": "graph_total_energy",
        "target": "total_energy_eV (graph-level)",
        "target_y_mean": float(y_all.mean().item()),
        "target_y_std": float(y_all.std(unbiased=False).item()),
        "target_y_min": float(y_all.min().item()),
        "target_y_max": float(y_all.max().item()),
        "num_atoms_min": int(min(n_atoms)),
        "num_atoms_median": int(np.median(n_atoms)),
        "num_atoms_max": int(max(n_atoms)),
        "materials_root": os.path.abspath(materials_root),
        "kinds": list(kinds),
        "notes": [
            "No per-atom PE in source data; initial-PE node feature omitted.",
            "is_defect / dist_to_defect are zero (no atom-level fault labels).",
            "y shape [1,1] = total potential energy from total_energy.txt.",
            "Optional attrs: cohesive_energy_eV, sfe_mjm2 when present.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(stats_path)) or ".", exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"Saved {len(augmented)} graphs -> {output_path}")
    print(f"Saved stats -> {stats_path}")
    print(
        f"  Target (total E): mean={stats['target_y_mean']:.4f}, "
        f"std={stats['target_y_std']:.4f}, "
        f"range=[{stats['target_y_min']:.4f}, {stats['target_y_max']:.4f}]"
    )
    return augmented, stats


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build cycle-augmented PyG dataset from 20261020 Ising/SFE data "
            "(OVITO relaxed.out + graph-level total_energy)."
        )
    )
    parser.add_argument(
        "--materials-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Path to to_send/materials",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-output", type=str, default=DEFAULT_STATS)
    parser.add_argument("--edge-k", type=int, default=EDGE_K)
    parser.add_argument("--edge-radius", type=float, default=3.0)
    parser.add_argument(
        "--edge-mode", type=str, default="shell", choices=("shell", "radius")
    )
    parser.add_argument(
        "--kinds",
        type=str,
        nargs="+",
        default=list(KINDS_DEFAULT),
        choices=("ising_polytype", "supercell", "reference"),
        help="Which structure families to include",
    )
    parser.add_argument(
        "--materials",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of material folder names (e.g. ac-ca ag-be)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of graphs (smoke tests)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    build_ising_sfe_dataset_with_cycles(
        materials_root=args.materials_root,
        output_path=args.output,
        stats_path=args.stats_output,
        edge_k=args.edge_k,
        edge_radius=args.edge_radius,
        edge_mode=args.edge_mode,
        kinds=args.kinds,
        materials=args.materials,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
