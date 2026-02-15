import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ovito.data import NearestNeighborFinder
from ovito.io import import_file
from torch_geometric.data import Data

DEFECT_CUTOFF_K = 8
EDGE_K = 3
DEFECT_CUTOFF_RADIUS = 6.0
EDGE_CUTOFF_RADIUS = 3.0
SHELL_TOL_REL = 0.02

FILENAME_RE = re.compile(
    r"^(relaxed|unrelaxed)_(\d+)-(\d+)-(\d+)-([^.]+)\.(data|dump)$"
)


def _parse_defect_filename(filename: str) -> Optional[Dict[str, str]]:
    match = FILENAME_RE.match(filename)
    if not match:
        return None
    return {
        "relax_state": match.group(1),
        "defect_id": int(match.group(2)),
        "from_type": int(match.group(3)),
        "to_type": int(match.group(4)),
        "wyckoff": match.group(5),
        "ext": match.group(6),
    }


def _get_property_array(particles, candidates: List[str]) -> Optional[np.ndarray]:
    for name in candidates:
        if name in particles:
            return np.asarray(particles[name])
    return None


def _get_particle_ids(data) -> np.ndarray:
    if data.particles.identifiers is not None and len(data.particles.identifiers) > 0:
        return np.asarray(data.particles.identifiers)
    if "Particle Identifier" in data.particles:
        return np.asarray(data.particles["Particle Identifier"])
    return np.arange(1, data.particles.count + 1, dtype=int)


def _load_types_from_data(data_path: str) -> Dict[int, float]:
    pipeline = import_file(data_path)
    data = pipeline.compute()
    particle_ids = _get_particle_ids(data)
    types = _get_property_array(data.particles, ["Particle Type", "Type", "type"])
    if types is None:
        raise ValueError(f"Missing particle type property in {data_path}")
    return {int(pid): float(types[idx]) for idx, pid in enumerate(particle_ids)}


def _build_subgraph(
    dump_path: str,
    relaxed_dump_path: str,
    data_path: str,
    defect_id: int,
    cutoff_k: int,
    edge_k: int,
    cutoff_radius: float,
    edge_radius: float,
    cutoff_mode: str,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[int, int],
    Dict,
]:
    pipeline = import_file(dump_path)
    data = pipeline.compute()

    particle_ids = _get_particle_ids(data)
    id_to_index = {pid: idx for idx, pid in enumerate(particle_ids)}
    if defect_id not in id_to_index:
        raise ValueError(f"Defect id {defect_id} not found in {dump_path}")
    defect_index = id_to_index[defect_id]

    positions = np.asarray(data.particles.positions)
    types = _get_property_array(data.particles, ["Particle Type", "Type", "type"])
    if types is None:
        type_by_id = _load_types_from_data(data_path)
        types = np.array(
            [type_by_id.get(int(pid), np.nan) for pid in particle_ids], dtype=float
        )
        if np.isnan(types).any():
            raise ValueError(f"Missing particle type mapping in {data_path}")
    per_atom_pe = _get_property_array(
        data.particles,
        ["c_pe_potential_energy", "c_pe_potential_energy[1]", "pe_potential_energy"],
    )
    if per_atom_pe is None:
        raise ValueError(f"Missing per-atom potential energy in {dump_path}")

    relaxed_pipeline = import_file(relaxed_dump_path)
    relaxed_data = relaxed_pipeline.compute()
    relaxed_particle_ids = _get_particle_ids(relaxed_data)
    relaxed_pe = _get_property_array(
        relaxed_data.particles,
        ["c_pe_potential_energy", "c_pe_potential_energy[1]", "pe_potential_energy"],
    )
    if relaxed_pe is None:
        raise ValueError(f"Missing per-atom potential energy in {relaxed_dump_path}")
    relaxed_pe_by_id = {
        int(pid): float(relaxed_pe[idx])
        for idx, pid in enumerate(relaxed_particle_ids)
    }

    cell = data.cell
    if cell is None:
        cell_matrix = np.eye(3)
        pbc = (False, False, False)
    else:
        cell_matrix = np.asarray(cell.matrix)
        if cell_matrix.ndim != 2 or cell_matrix.shape[0] != 3:
            cell_matrix = np.eye(3)
            pbc = (False, False, False)
        else:
            if cell_matrix.shape[1] > 3:
                cell_matrix = cell_matrix[:, :3]
            pbc = tuple(bool(v) for v in cell.pbc)

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

    tol = 1e-6
    all_dist_to_defect = np.zeros(len(positions), dtype=float)
    for idx in range(len(positions)):
        if idx == defect_index:
            all_dist_to_defect[idx] = 0.0
        else:
            dvec = _min_image_delta(defect_index, idx)
            all_dist_to_defect[idx] = float(np.linalg.norm(dvec))

    def _shell_threshold(sorted_distances: np.ndarray, k_shells: int) -> float:
        base_dist = sorted_distances[0]
        shell_tol = max(base_dist * SHELL_TOL_REL, 1e-6)
        shell_distances = [sorted_distances[0]]
        for dist in sorted_distances[1:]:
            if abs(dist - shell_distances[-1]) > shell_tol:
                shell_distances.append(dist)
        cutoff_idx = min(k_shells, len(shell_distances)) - 1
        return shell_distances[cutoff_idx]

    if cutoff_mode == "shell":
        non_self = np.array([d for d in all_dist_to_defect if d > 0.0])
        if len(non_self) > 0:
            sorted_dist = np.sort(non_self)
            cutoff_dist = _shell_threshold(sorted_dist, cutoff_k)
        else:
            cutoff_dist = 0.0
    elif cutoff_mode == "radius":
        cutoff_dist = float(cutoff_radius)
    else:
        raise ValueError(f"Unsupported cutoff_mode: {cutoff_mode}")

    subset_indices = [
        idx
        for idx, dist in enumerate(all_dist_to_defect)
        if dist <= cutoff_dist + tol
    ]
    if defect_index not in subset_indices:
        subset_indices.append(defect_index)
    subset_indices = sorted(set(subset_indices))
    dist_to_defect = {idx: all_dist_to_defect[idx] for idx in subset_indices}

    subset_indices = sorted(subset_indices)
    sub_index = {orig: i for i, orig in enumerate(subset_indices)}

    node_features = []
    for orig in subset_indices:
        node_features.append(
            [
                float(types[orig]),
                float(per_atom_pe[orig]),
                1.0 if orig == defect_index else 0.0,
                float(dist_to_defect.get(orig, 0.0)),
            ]
        )

    x = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(positions[subset_indices], dtype=torch.float)
    relaxed_targets = []
    for orig in subset_indices:
        pid = int(particle_ids[orig])
        if pid not in relaxed_pe_by_id:
            raise ValueError(f"Missing relaxed per-atom PE for particle id {pid}")
        relaxed_targets.append([relaxed_pe_by_id[pid]])
    y_node = torch.tensor(relaxed_targets, dtype=torch.float)

    subset_count = len(subset_indices)
    dist_matrix = np.zeros((subset_count, subset_count), dtype=float)
    for i in range(subset_count):
        for j in range(i + 1, subset_count):
            dvec = _min_image_delta(subset_indices[i], subset_indices[j])
            dist = float(np.linalg.norm(dvec))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    edge_index: List[List[int]] = []
    edge_attr: List[List[float]] = []
    edge_set = set()
    for i in range(subset_count):
        distances = dist_matrix[i]
        neighbor_dists = [d for d in distances if d > 0.0]
        if not neighbor_dists:
            continue
        sorted_dist = np.sort(np.array(neighbor_dists))
        if cutoff_mode == "shell":
            edge_dist = _shell_threshold(sorted_dist, edge_k)
        elif cutoff_mode == "radius":
            edge_dist = float(edge_radius)
        else:
            raise ValueError(f"Unsupported cutoff_mode: {cutoff_mode}")
        for j in range(subset_count):
            if i == j:
                continue
            if distances[j] <= edge_dist:
                src = i
                dst = j
                key = (min(src, dst), max(src, dst))
                if key in edge_set:
                    continue
                edge_set.add(key)
                edge_index.append([src, dst])
                same_type = (
                    1.0
                    if types[subset_indices[i]] == types[subset_indices[j]]
                    else 0.0
                )
                incident_defect = 1.0 if (
                    subset_indices[i] == defect_index
                    or subset_indices[j] == defect_index
                ) else 0.0
                edge_attr.append(
                    [float(distances[j]), same_type, incident_defect]
                )

    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.zeros((0, 3), dtype=torch.float)

    meta = {
        "defect_index": defect_index,
        "subset_size": len(subset_indices),
        "cutoff_distance": float(cutoff_dist),
        "cutoff_mode": cutoff_mode,
    }
    return x, pos, edge_index_tensor, edge_attr_tensor, y_node, sub_index, meta


def build_pyg_dataset(
    simulations_dir: str,
    cutoff_k: int = DEFECT_CUTOFF_K,
    edge_k: int = EDGE_K,
    cutoff_radius: float = DEFECT_CUTOFF_RADIUS,
    edge_radius: float = EDGE_CUTOFF_RADIUS,
    cutoff_mode: str = "shell",
) -> List[Data]:
    dataset: List[Data] = []

    for folder in sorted(os.listdir(simulations_dir)):
        if folder.endswith("_MIN"):
            continue
        folder_path = os.path.join(simulations_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        file_count = len(
            [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        )
        if file_count != 52:
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

            x, pos, edge_index, edge_attr, y_node, sub_index, meta = _build_subgraph(
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

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                y=y_node,
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
            data.meta = meta
            dataset.append(data)

    return dataset


def save_dataset(dataset: List[Data], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(dataset, output_path)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    simulations = os.path.join(root_dir, "SIMULATIONS")
    output_file = os.path.join(root_dir, "pyg_dataset.pt")

    pyg_dataset = build_pyg_dataset(
        simulations_dir=simulations,
        cutoff_k=DEFECT_CUTOFF_K,
        edge_k=EDGE_K,
    )
    save_dataset(pyg_dataset, output_file)

    print(f"Saved {len(pyg_dataset)} graphs to {output_file}")