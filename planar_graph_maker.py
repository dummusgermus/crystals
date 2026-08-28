"""Build PyG datasets from planar / Laves stacking simulations.

Paired archives (default — same ML task as :mod:`graph_maker`):

* **Initial PE** from ``Laves_Planar_Defects/SIMULATIONS/*/minimised.dump``
  (pe/atom at the unrelaxed stack geometry).
* **Relaxed PE** from ``Laves_Screen/SIMULATIONS/*/minimised.dump``
  (z-box + ionic minimisation of the same stacks).
* **Geometry** from ``basefile.data`` (prefer the initial campaign folder).
* **Defect labels** default to C14/C15 deviation
  (``laves_defect_atoms_c14c15.json``).

Unlike :mod:`graph_maker`, every atom in the supercell is kept — no
defect-centred subgraph cutoff — but **edge wiring uses the same k-shell
neighbour logic** as the bulk defect pipeline.

Node features (same schema as :mod:`graph_maker`):

  ``[type, per_atom_pe, is_defect, dist_to_defect]`` + normalised 3/4-cycles

* ``per_atom_pe`` — initial (unrelaxed) pe/atom, as in the point-defect path.
* ``y`` — **absolute relaxed** pe/atom from the Screen campaign, matching
  :mod:`graph_maker` (not the residual).

Outputs ``planar_pyg_dataset_c14c15.pt`` (and matching stats JSON) by default.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

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

PLANAR_FOLDER_RE = re.compile(r"^([A-Z][a-z]?)-([A-Z][a-z]?)_(.+)$")

UNRELAXED_PE_CANDIDATES = [
    "c_pe_potential_energy",
    "c_pe_potential_energy[1]",
    "pe_potential_energy",
    "c_pe",
]
RELAXED_PE_CANDIDATES = list(UNRELAXED_PE_CANDIDATES)

UNRELAXED_DUMP_CANDIDATES = (
    "unrelaxed.dump",
    "initial.dump",
    "pre_relax.dump",
)

DEFECT_ID_FILE_CANDIDATES = (
    "defect_id.txt",
    "stack_defect_id.txt",
    "replace_id.txt",
)

DEFECT_YAML_CANDIDATES = (
    "basefile.yaml",
    "defect.yaml",
    "simulation.yaml",
)

DEFECT_YAML_KEYS = (
    "defect_id",
    "stack_defect_id",
    "defect_atom_id",
)

# Best-performing planar defect definition from the label ablation.
DEFECT_ATOMS_JSON = "laves_defect_atoms_c14c15.json"
# Legacy broad ISF/ESF mapping (kept for re-runs / comparison).
DEFECT_ATOMS_JSON_BASELINE = "laves_defect_atoms.json"

CYCLE_COLS = (0, 1)  # 3- and 4-cycles (cycle34 variant)
CYCLE_LENGTHS = (3, 4)

TARGET_MODES = ("residual", "absolute")


def load_defect_atoms_by_stack(
    root_dir: Optional[str] = None,
    json_path: Optional[str] = None,
) -> Dict[str, List[int]]:
    """Load the composition-independent planar-fault atom mapping.

    Returns a ``{stack_sequence: [atom_id, ...]}`` dict.  By default reads
    ``laves_defect_atoms.json`` next to this module.  Pass *json_path* to use
    an alternate definition file (e.g. narrowed C14/C15 labels).  Returns an
    empty dict if the file is missing so graph building still works (without
    defect labels).  Empty id lists are preserved — they mean “explicitly no
    defect atoms for this stack”, which is distinct from a missing key.
    """
    if json_path is None:
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(root_dir, DEFECT_ATOMS_JSON)
    if not os.path.isfile(json_path):
        return {}
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    mapping = payload.get("defect_atoms", payload)
    return {
        str(stack): [int(a) for a in ids]
        for stack, ids in mapping.items()
        if not str(stack).startswith("_")
    }


def _is_nonempty_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def folder_has_complete_data(folder_path: str) -> bool:
    """Return True when a folder has the minimum files needed for graph building."""
    return (
        _is_nonempty_file(os.path.join(folder_path, "basefile.data"))
        and _is_nonempty_file(os.path.join(folder_path, "minimised.dump"))
    )


def folder_has_dump(folder_path: str) -> bool:
    """Return True when *folder_path* has a non-empty ``minimised.dump``."""
    return _is_nonempty_file(os.path.join(folder_path, "minimised.dump"))


def resolve_unrelaxed_dump(folder_path: str) -> Optional[str]:
    """Return the first existing pre-relax dump path in *folder_path*, if any.

    Only checks explicit unrelaxed dump names.  For the
    ``Laves_Planar_Defects`` campaign (relax off, pe/atom in
    ``minimised.dump``), use :func:`resolve_paired_paths` instead — do not
    treat a Screen ``minimised.dump`` as initial PE.
    """
    for name in UNRELAXED_DUMP_CANDIDATES:
        path = os.path.join(folder_path, name)
        if _is_nonempty_file(path):
            return path
    return None


def resolve_paired_paths(
    folder: str,
    initial_dir: str,
    relaxed_dir: str,
) -> Optional[Tuple[str, str, str]]:
    """Resolve ``(basefile, initial_dump, relaxed_dump)`` for a paired folder.

    Returns ``None`` when either archive is missing required files.
    Initial PE is read from ``initial_dir/.../minimised.dump`` (no-relax
    pe/atom campaign) or from an explicit ``unrelaxed.dump`` if present.
    """
    initial_folder = os.path.join(initial_dir, folder)
    relaxed_folder = os.path.join(relaxed_dir, folder)

    if not os.path.isdir(initial_folder) or not os.path.isdir(relaxed_folder):
        return None

    # Prefer geometry from the initial campaign; fall back to Screen basefile.
    basefile = os.path.join(initial_folder, "basefile.data")
    if not _is_nonempty_file(basefile):
        basefile = os.path.join(relaxed_folder, "basefile.data")
    if not _is_nonempty_file(basefile):
        return None

    initial_dump = resolve_unrelaxed_dump(initial_folder)
    if initial_dump is None:
        # Laves_Planar_Defects: no-relax pe/atom run writes minimised.dump.
        candidate = os.path.join(initial_folder, "minimised.dump")
        if _is_nonempty_file(candidate):
            initial_dump = candidate
    if initial_dump is None:
        return None

    relaxed_dump = os.path.join(relaxed_folder, "minimised.dump")
    if not _is_nonempty_file(relaxed_dump):
        return None

    # Guard against accidentally pointing both dirs at the same campaign.
    if os.path.normcase(os.path.abspath(initial_dump)) == os.path.normcase(
        os.path.abspath(relaxed_dump)
    ):
        return None

    return basefile, initial_dump, relaxed_dump


def _parse_yaml_mapping(path: str) -> Optional[dict]:
    try:
        from ruamel.yaml import YAML
    except ImportError:
        return None

    yaml = YAML()
    with open(path, encoding="utf-8") as fh:
        data = yaml.load(fh)
    if isinstance(data, list):
        data = data[0] if data else None
    return data if isinstance(data, dict) else None


def load_defect_ids(
    folder_path: str,
    stack: Optional[str] = None,
    defect_atoms_by_stack: Optional[Dict[str, List[int]]] = None,
) -> Optional[List[int]]:
    """Return the planar-fault atom ids for a folder, or ``None`` if unknown.

    Resolution order:

    1. Per-folder sidecar files listing one or more ids (``defect_id.txt`` …).
    2. Per-folder YAML keys (``defect_id`` …), scalar or list.
    3. The composition-independent stack-sequence mapping
       (``laves_defect_atoms.json``).
    """
    for name in DEFECT_ID_FILE_CANDIDATES:
        path = os.path.join(folder_path, name)
        if not _is_nonempty_file(path):
            continue
        with open(path, encoding="utf-8") as fh:
            tokens = fh.read().replace(",", " ").split()
        ids = [int(tok) for tok in tokens if tok]
        if ids:
            return ids

    for name in DEFECT_YAML_CANDIDATES:
        path = os.path.join(folder_path, name)
        if not _is_nonempty_file(path):
            continue
        mapping = _parse_yaml_mapping(path)
        if not mapping:
            continue
        for key in DEFECT_YAML_KEYS:
            if key in mapping and mapping[key] is not None:
                value = mapping[key]
                if isinstance(value, (list, tuple)):
                    return [int(v) for v in value]
                return [int(value)]

    if (
        stack is not None
        and defect_atoms_by_stack is not None
        and stack in defect_atoms_by_stack
    ):
        # Preserve empty lists: they mean "known stack, no defect atoms".
        return list(defect_atoms_by_stack[stack])
    return None


def parse_planar_folder(folder: str) -> Tuple[str, str, str]:
    """Return ``(element_a, element_b, stack_sequence)`` from folder name."""
    m = PLANAR_FOLDER_RE.match(folder)
    if not m:
        raise ValueError(
            f"Could not parse planar folder name {folder!r}; "
            "expected format like Ag-Be_XY'XY"
        )
    elem_a, elem_b, stack = m.group(1), m.group(2), m.group(3)
    for sym in (elem_a, elem_b):
        if sym not in SYMBOL_TO_Z:
            raise ValueError(
                f"Folder {folder!r} references element {sym!r} "
                "outside the H..Fm vocabulary."
            )
    return elem_a, elem_b, stack


def build_type_to_z_map(elem_a: str, elem_b: str) -> Dict[int, int]:
    """Map LAMMPS type 1/2 to atomic numbers (same order as ``pair_coeff``)."""
    return {1: SYMBOL_TO_Z[elem_a], 2: SYMBOL_TO_Z[elem_b]}


def _shell_threshold(sorted_distances: np.ndarray, k_shells: int) -> float:
    if len(sorted_distances) == 0:
        return 0.0
    base_dist = sorted_distances[0]
    shell_tol = max(base_dist * SHELL_TOL_REL, 1e-6)
    shell_distances = [sorted_distances[0]]
    for dist in sorted_distances[1:]:
        if abs(dist - shell_distances[-1]) > shell_tol:
            shell_distances.append(dist)
    cutoff_idx = min(k_shells, len(shell_distances)) - 1
    return shell_distances[cutoff_idx]


def _cell_and_pbc(data) -> Tuple[np.ndarray, Tuple[bool, bool, bool]]:
    cell = data.cell
    if cell is None:
        return np.eye(3), (False, False, False)
    cell_matrix = np.asarray(cell.matrix)
    if cell_matrix.ndim != 2 or cell_matrix.shape[0] != 3:
        return np.eye(3), (False, False, False)
    if cell_matrix.shape[1] > 3:
        cell_matrix = cell_matrix[:, :3]
    return cell_matrix, tuple(bool(v) for v in cell.pbc)


def _build_full_graph(
    basefile_path: str,
    relaxed_dump_path: str,
    unrelaxed_dump_path: Optional[str],
    defect_ids: Optional[List[int]],
    edge_k: int,
    edge_radius: float,
    edge_mode: str,
    target_mode: str = "absolute",
    require_initial_pe: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict,
    torch.Tensor,
]:
    """Build one full-cell graph from unrelaxed geometry and relaxed targets.

    *target_mode* ``residual`` sets ``y = PE_relaxed − PE_initial``;
    ``absolute`` sets ``y = PE_relaxed``.

    Returns ``particle_ids`` aligned with node order (LAMMPS ids).
    """
    if target_mode not in TARGET_MODES:
        raise ValueError(
            f"Unsupported target_mode {target_mode!r}; "
            f"expected one of {TARGET_MODES}"
        )

    from ovito.io import import_file

    base = import_file(basefile_path).compute()
    relaxed = import_file(relaxed_dump_path).compute()

    particle_ids = _get_particle_ids(base)
    relaxed_ids = _get_particle_ids(relaxed)
    id_to_index = {int(pid): idx for idx, pid in enumerate(particle_ids)}

    types = _get_property_array(
        base.particles, ["Particle Type", "Type", "type"]
    )
    if types is None:
        raise ValueError(f"Missing particle types in {basefile_path}")

    positions = np.asarray(base.particles.positions)
    n_atoms = len(positions)

    per_atom_pe = np.zeros(n_atoms, dtype=float)
    has_unrelaxed_pe = False
    if unrelaxed_dump_path and os.path.exists(unrelaxed_dump_path):
        unrelaxed = import_file(unrelaxed_dump_path).compute()
        unrelaxed_ids = _get_particle_ids(unrelaxed)
        unrelaxed_pe = _get_property_array(unrelaxed.particles, UNRELAXED_PE_CANDIDATES)
        if unrelaxed_pe is not None:
            pe_by_id = {
                int(pid): float(unrelaxed_pe[idx])
                for idx, pid in enumerate(unrelaxed_ids)
            }
            missing_ids = [
                int(pid) for pid in particle_ids if int(pid) not in pe_by_id
            ]
            if missing_ids:
                raise ValueError(
                    f"Missing initial per-atom PE for particle ids "
                    f"{missing_ids[:5]}{'…' if len(missing_ids) > 5 else ''} "
                    f"in {unrelaxed_dump_path}"
                )
            for idx, pid in enumerate(particle_ids):
                per_atom_pe[idx] = pe_by_id[int(pid)]
            has_unrelaxed_pe = True

    if require_initial_pe and not has_unrelaxed_pe:
        raise ValueError(
            f"Initial per-atom PE required but not found "
            f"(looked for dump at {unrelaxed_dump_path!r})"
        )
    if target_mode == "residual" and not has_unrelaxed_pe:
        raise ValueError(
            "target_mode='residual' requires initial per-atom PE"
        )

    relaxed_pe = _get_property_array(relaxed.particles, RELAXED_PE_CANDIDATES)
    if relaxed_pe is None:
        raise ValueError(f"Missing relaxed per-atom PE in {relaxed_dump_path}")
    relaxed_pe_by_id = {
        int(pid): float(relaxed_pe[idx]) for idx, pid in enumerate(relaxed_ids)
    }

    cell_matrix, pbc = _cell_and_pbc(base)
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

    defect_indices: set = set()
    has_defect_labels = False
    dist_to_defect = np.zeros(n_atoms, dtype=float)
    if defect_ids:
        missing = [d for d in defect_ids if d not in id_to_index]
        if missing:
            raise ValueError(
                f"Defect ids {missing} not found in {basefile_path}"
            )
        defect_indices = {id_to_index[d] for d in defect_ids}
        has_defect_labels = True
        for idx in range(n_atoms):
            if idx in defect_indices:
                dist_to_defect[idx] = 0.0
            else:
                dist_to_defect[idx] = min(
                    float(np.linalg.norm(_min_image_delta(d_idx, idx)))
                    for d_idx in defect_indices
                )

    node_features: List[List[float]] = []
    targets: List[List[float]] = []
    for idx in range(n_atoms):
        pid = int(particle_ids[idx])
        if pid not in relaxed_pe_by_id:
            raise ValueError(
                f"Missing relaxed per-atom PE for particle id {pid} "
                f"in {relaxed_dump_path}"
            )
        pe_relaxed = relaxed_pe_by_id[pid]
        pe_initial = float(per_atom_pe[idx])
        if target_mode == "residual":
            y_val = pe_relaxed - pe_initial
        else:
            y_val = pe_relaxed
        node_features.append(
            [
                float(types[idx]),
                pe_initial,
                1.0 if idx in defect_indices else 0.0,
                float(dist_to_defect[idx]),
            ]
        )
        targets.append([y_val])

    x = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(positions, dtype=torch.float)
    y_node = torch.tensor(targets, dtype=torch.float)

    pe_initial_total = float(np.sum(per_atom_pe))
    pe_true_total = pe_initial_total + float(y_node.sum().item())
    delta_total = pe_true_total - pe_initial_total
    graph_delta_total = float(y_node.sum().item())

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
        "has_unrelaxed_pe": has_unrelaxed_pe,
        "has_defect_labels": has_defect_labels,
        "num_defect_atoms": len(defect_indices),
        "defect_ids": list(defect_ids) if defect_ids else [],
        "target_mode": target_mode,
        "pe_initial_total_eV": pe_initial_total,
        "pe_true_total_eV": pe_true_total,
        "delta_total_eV": delta_total,
        "graph_delta_total_eV": graph_delta_total,
    }
    particle_ids_tensor = torch.tensor(
        [int(pid) for pid in particle_ids],
        dtype=torch.long,
    )
    return x, pos, edge_index_tensor, edge_attr_tensor, y_node, meta, particle_ids_tensor


def build_planar_pyg_dataset(
    simulations_dir: Optional[str] = None,
    edge_k: int = EDGE_K,
    edge_radius: float = 3.0,
    edge_mode: str = "shell",
    defect_atoms_json: Optional[str] = None,
    initial_simulations_dir: Optional[str] = None,
    relaxed_simulations_dir: Optional[str] = None,
    target_mode: str = "absolute",
    require_initial_pe: bool = True,
) -> List[Data]:
    """Build full-cell graphs for every valid planar simulation folder.

    Preferred mode (paired archives):

    * ``initial_simulations_dir`` — ``Laves_Planar_Defects/SIMULATIONS``
      (initial pe/atom in ``minimised.dump``)
    * ``relaxed_simulations_dir`` — ``Laves_Screen/SIMULATIONS``
      (relaxed pe/atom in ``minimised.dump``)

    Legacy single-directory mode: pass ``simulations_dir`` only (same folder
    supplies geometry + relaxed dump; initial PE from ``unrelaxed.dump`` etc.
    when present).
    """
    if target_mode not in TARGET_MODES:
        raise ValueError(
            f"Unsupported target_mode {target_mode!r}; "
            f"expected one of {TARGET_MODES}"
        )

    paired = initial_simulations_dir is not None or relaxed_simulations_dir is not None
    if paired:
        if not initial_simulations_dir or not relaxed_simulations_dir:
            raise ValueError(
                "Both initial_simulations_dir and relaxed_simulations_dir "
                "must be set for paired archive mode."
            )
        if not os.path.isdir(initial_simulations_dir):
            raise FileNotFoundError(
                f"initial_simulations_dir not found: {initial_simulations_dir}"
            )
        if not os.path.isdir(relaxed_simulations_dir):
            raise FileNotFoundError(
                f"relaxed_simulations_dir not found: {relaxed_simulations_dir}"
            )
        folder_source = relaxed_simulations_dir
    else:
        if not simulations_dir:
            raise ValueError(
                "Pass simulations_dir, or both initial_simulations_dir and "
                "relaxed_simulations_dir."
            )
        if not os.path.isdir(simulations_dir):
            raise FileNotFoundError(f"simulations_dir not found: {simulations_dir}")
        folder_source = simulations_dir

    dataset: List[Data] = []
    skipped_incomplete = 0

    defect_atoms_by_stack = load_defect_atoms_by_stack(json_path=defect_atoms_json)
    if defect_atoms_by_stack:
        labeled = sum(1 for ids in defect_atoms_by_stack.values() if ids)
        print(
            f"  Loaded planar-fault atom mapping for "
            f"{len(defect_atoms_by_stack)} stack sequences "
            f"({labeled} with >=1 defect atom)"
            + (f" from {defect_atoms_json}" if defect_atoms_json else "")
            + "."
        )
    else:
        print(
            "  [warn] No planar-fault atom mapping found; "
            "is_defect / dist_to_defect will be zero."
        )

    print(f"  target_mode={target_mode}, require_initial_pe={require_initial_pe}")
    if paired:
        print(f"  initial PE dir: {os.path.abspath(initial_simulations_dir)}")
        print(f"  relaxed PE dir: {os.path.abspath(relaxed_simulations_dir)}")
    else:
        print(f"  simulations dir: {os.path.abspath(simulations_dir)}")

    for folder in sorted(os.listdir(folder_source)):
        folder_path = os.path.join(folder_source, folder)
        if not os.path.isdir(folder_path):
            continue

        if paired:
            resolved = resolve_paired_paths(
                folder, initial_simulations_dir, relaxed_simulations_dir
            )
            if resolved is None:
                skipped_incomplete += 1
                continue
            basefile_path, unrelaxed_dump_path, relaxed_dump_path = resolved
            label_folder = os.path.join(initial_simulations_dir, folder)
            if not os.path.isdir(label_folder):
                label_folder = folder_path
        else:
            if not folder_has_complete_data(folder_path):
                skipped_incomplete += 1
                continue
            basefile_path = os.path.join(folder_path, "basefile.data")
            relaxed_dump_path = os.path.join(folder_path, "minimised.dump")
            unrelaxed_dump_path = resolve_unrelaxed_dump(folder_path)
            label_folder = folder_path

        try:
            elem_a, elem_b, stack = parse_planar_folder(folder)
            type_to_z = build_type_to_z_map(elem_a, elem_b)
        except ValueError as err:
            print(f"  [skip] {folder}: {err}")
            continue

        defect_ids = load_defect_ids(label_folder, stack, defect_atoms_by_stack)
        try:
            x, pos, edge_index, edge_attr, y_node, meta, particle_ids = _build_full_graph(
                basefile_path=basefile_path,
                relaxed_dump_path=relaxed_dump_path,
                unrelaxed_dump_path=unrelaxed_dump_path,
                defect_ids=defect_ids,
                edge_k=edge_k,
                edge_radius=edge_radius,
                edge_mode=edge_mode,
                target_mode=target_mode,
                require_initial_pe=require_initial_pe,
            )
        except (ValueError, OSError) as err:
            print(f"  [skip] {folder}: {err}")
            continue

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
        data.folder = folder
        data.element_a = elem_a
        data.element_b = elem_b
        data.stack_sequence = stack
        data.edge_k = edge_k
        data.edge_radius = edge_radius
        data.edge_mode = edge_mode
        data.target_mode = target_mode
        data.meta = meta
        data.particle_ids = particle_ids
        data.delta_total_eV = torch.tensor([meta["delta_total_eV"]], dtype=torch.float)
        if defect_ids is not None:
            data.defect_ids = list(defect_ids)
        dataset.append(data)

    if skipped_incomplete:
        print(
            f"  Skipped {skipped_incomplete} folders without complete paired "
            "simulation output (missing basefile / initial dump / relaxed dump)."
        )

    return dataset


def build_planar_dataset_with_cycles(
    simulations_dir: Optional[str] = None,
    output_path: str = "",
    stats_path: str = "",
    edge_k: int = EDGE_K,
    edge_radius: float = 3.0,
    edge_mode: str = "shell",
    defect_atoms_json: Optional[str] = None,
    initial_simulations_dir: Optional[str] = None,
    relaxed_simulations_dir: Optional[str] = None,
    target_mode: str = "absolute",
    require_initial_pe: bool = True,
) -> Tuple[List[Data], Dict]:
    """Build graphs, append normalised 3/4-cycle features, and save."""
    print("Building planar base dataset …")
    t0 = time.time()
    dataset = build_planar_pyg_dataset(
        simulations_dir=simulations_dir,
        edge_k=edge_k,
        edge_radius=edge_radius,
        edge_mode=edge_mode,
        defect_atoms_json=defect_atoms_json,
        initial_simulations_dir=initial_simulations_dir,
        relaxed_simulations_dir=relaxed_simulations_dir,
        target_mode=target_mode,
        require_initial_pe=require_initial_pe,
    )
    if not dataset:
        raise RuntimeError(
            "No graphs built. Expected paired folders with basefile.data, "
            "initial pe/atom dump (Laves_Planar_Defects), and relaxed "
            "minimised.dump (Laves_Screen)."
        )
    print(f"  {len(dataset)} graphs built in {time.time() - t0:.1f}s")

    print("Computing per-node 3/4-cycle counts …")
    t0 = time.time()
    all_cycle_feats: List[np.ndarray] = []
    for i, graph in enumerate(dataset):
        feats = count_cycles_per_node(graph.edge_index, graph.num_nodes, max_cycle_len=4)
        all_cycle_feats.append(feats)
        if (i + 1) % 200 == 0 or i + 1 == len(dataset):
            print(f"  [{i + 1}/{len(dataset)}] graphs processed")
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

    y_all = torch.cat([g.y for g in augmented], dim=0).view(-1)
    default_json = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), DEFECT_ATOMS_JSON
    )
    stats = {
        "cycle_lengths": list(CYCLE_LENGTHS),
        "mean": v_mean.tolist(),
        "std": v_std.tolist(),
        "num_graphs": len(augmented),
        "base_feature_dim": int(dataset[0].x.size(-1)),
        "total_feature_dim": int(augmented[0].x.size(-1)),
        "node_features": [
            "type",
            "per_atom_pe",
            "is_defect",
            "dist_to_defect",
            "cycle3_norm",
            "cycle4_norm",
        ],
        "edge_features": ["distance", "same_type", "incident_defect"],
        "edge_k": edge_k,
        "edge_mode": edge_mode,
        "target_mode": target_mode,
        "target_y_mean": float(y_all.mean().item()),
        "target_y_std": float(y_all.std(unbiased=False).item()),
        "target_y_min": float(y_all.min().item()),
        "target_y_max": float(y_all.max().item()),
        "simulations_dir": (
            os.path.abspath(simulations_dir) if simulations_dir else None
        ),
        "initial_simulations_dir": (
            os.path.abspath(initial_simulations_dir)
            if initial_simulations_dir
            else None
        ),
        "relaxed_simulations_dir": (
            os.path.abspath(relaxed_simulations_dir)
            if relaxed_simulations_dir
            else None
        ),
        "defect_atoms_json": (
            os.path.abspath(defect_atoms_json)
            if defect_atoms_json
            else os.path.abspath(default_json)
        ),
        "graphs_with_unrelaxed_pe": sum(
            1 for g in dataset if g.meta.get("has_unrelaxed_pe", False)
        ),
        "graphs_with_defect_labels": sum(
            1 for g in dataset if g.meta.get("has_defect_labels", False)
        ),
        "total_defect_atoms": sum(
            g.meta.get("num_defect_atoms", 0) for g in dataset
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(stats_path)) or ".", exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"Saved {len(augmented)} graphs -> {output_path}")
    print(f"Saved stats -> {stats_path}")
    print(
        f"  Target ({target_mode}): mean={stats['target_y_mean']:.6f}, "
        f"std={stats['target_y_std']:.6f}, "
        f"range=[{stats['target_y_min']:.6f}, {stats['target_y_max']:.6f}]"
    )
    print(
        f"  Graphs with initial PE: {stats['graphs_with_unrelaxed_pe']}/{len(augmented)}"
    )
    return augmented, stats


if __name__ == "__main__":
    import argparse

    root_dir = os.path.dirname(os.path.abspath(__file__))
    default_initial = os.path.join(root_dir, "Laves_Planar_Defects", "SIMULATIONS")
    default_relaxed = os.path.join(root_dir, "Laves_Screen", "SIMULATIONS")
    default_defect_json = os.path.join(root_dir, DEFECT_ATOMS_JSON)
    default_output = os.path.join(root_dir, "planar_pyg_dataset_c14c15.pt")
    default_stats = os.path.join(
        root_dir, "planar_pyg_dataset_c14c15_stats.json"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Build a cycle-augmented PyG dataset from planar Laves stacking "
            "simulations. Default: pair Laves_Planar_Defects (initial PE) with "
            "Laves_Screen (relaxed PE), predict absolute relaxed PE, C14/C15 labels."
        )
    )
    parser.add_argument(
        "--initial-simulations-dir",
        type=str,
        default=default_initial,
        help="Archive with initial pe/atom (Laves_Planar_Defects/SIMULATIONS).",
    )
    parser.add_argument(
        "--relaxed-simulations-dir",
        type=str,
        default=default_relaxed,
        help="Archive with relaxed pe/atom (Laves_Screen/SIMULATIONS).",
    )
    parser.add_argument(
        "--simulations-dir",
        type=str,
        default=None,
        help=(
            "Legacy single-directory mode (geometry + relaxed dump in one tree). "
            "If set, disables paired-archive defaults."
        ),
    )
    parser.add_argument("--output", type=str, default=default_output)
    parser.add_argument("--stats-output", type=str, default=default_stats)
    parser.add_argument("--edge-k", type=int, default=EDGE_K)
    parser.add_argument("--edge-radius", type=float, default=3.0)
    parser.add_argument(
        "--edge-mode",
        choices=["shell", "radius"],
        default="shell",
        help="Neighbour shell for edge wiring (same semantics as graph_maker).",
    )
    parser.add_argument(
        "--defect-atoms-json",
        type=str,
        default=default_defect_json,
        help=(
            "Stack→defect-atom mapping JSON "
            f"(default: {DEFECT_ATOMS_JSON})."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=list(TARGET_MODES),
        default="absolute",
        help="Predict absolute relaxed PE (default, matches graph_maker) or residual ΔPE.",
    )
    parser.add_argument(
        "--allow-missing-initial-pe",
        action="store_true",
        help="Do not require initial pe/atom (legacy absolute-only builds).",
    )
    args = parser.parse_args()

    if args.simulations_dir:
        build_planar_dataset_with_cycles(
            simulations_dir=args.simulations_dir,
            output_path=args.output,
            stats_path=args.stats_output,
            edge_k=args.edge_k,
            edge_radius=args.edge_radius,
            edge_mode=args.edge_mode,
            defect_atoms_json=args.defect_atoms_json,
            target_mode=args.target_mode,
            require_initial_pe=not args.allow_missing_initial_pe,
        )
    else:
        build_planar_dataset_with_cycles(
            initial_simulations_dir=args.initial_simulations_dir,
            relaxed_simulations_dir=args.relaxed_simulations_dir,
            output_path=args.output,
            stats_path=args.stats_output,
            edge_k=args.edge_k,
            edge_radius=args.edge_radius,
            edge_mode=args.edge_mode,
            defect_atoms_json=args.defect_atoms_json,
            target_mode=args.target_mode,
            require_initial_pe=not args.allow_missing_initial_pe,
        )
    print("Done.")
