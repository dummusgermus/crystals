"""Export per-atom absolute PE predictions into mirrored crystal folders.

Modes (``--mode``):

* ``export``: full pipeline — dumps + model → per-architecture CSV/timing.
* ``inference``: model only (no dumps). Writes ``*_graph_pred.npz`` +
  ``*_timing.json`` (cluster-friendly; ``.npz`` avoids ``*.pt`` gitignore).
* ``restore``: dumps + one architecture's preds → CSV.
* ``restore-merged``: dumps + CGCNN+Transformer preds → one CSV per crystal
  under ``point/`` or ``planar/``, with both models' runtimes in the header.

Merged CSV columns::

    atom_id,pe_initial,pe_true,pe_pred_cgcnn,pe_pred_transformer[,in_graph]

``pe_pred_*`` is absolute PE via ``PE_initial + ΔPE_pred``.  Point-defect atoms
outside the subgraph keep ``ΔPE = 0``.

Cluster inference (no dumps)::

    python crystal_prediction_export.py --mode inference --job all \\
        --output-root predictions_inference

Local merged restore (cluster timings + local dumps)::

    python crystal_prediction_export.py --mode restore-merged --job all \\
        --inference-root predictions_inference --output-root predictions
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_models import (
    build_gated_model_from_dataset,
    build_graph_transformer_from_dataset,
)
from train_single import within_group_train_val_indices

ROOT = os.path.dirname(os.path.abspath(__file__))

# Must match residual delivery training (see curves JSONs / SLURM).
DELIVERY_SPLIT_SEED = 42
DELIVERY_VAL_FRACTION = 0.1

# Keep in sync with graph_maker / planar_graph_maker.
DEFECT_CUTOFF_K = 8
DEFECT_CUTOFF_RADIUS = 6.0
SHELL_TOL_REL = 0.02

POINT_PE_CANDIDATES = [
    "c_pe_potential_energy",
    "c_pe_potential_energy[1]",
    "pe_potential_energy",
    "c_pe",
]
PLANAR_PE_CANDIDATES = list(POINT_PE_CANDIDATES)

JOBS: Dict[str, Dict[str, str]] = {
    "point_cgcnn": {
        "domain": "point",
        "architecture": "cgcnn",
        "dataset": os.path.join(ROOT, "adv_datasets", "cycle34_residual_dataset.pt"),
        "checkpoint": os.path.join(ROOT, "cgcnn_defect_residual_model.pt"),
        "simulations_dir": os.path.join(ROOT, "SIMULATIONS"),
        "output_subdir": "point_cgcnn",
    },
    "point_transformer": {
        "domain": "point",
        "architecture": "transformer",
        "dataset": os.path.join(ROOT, "adv_datasets", "cycle34_residual_dataset.pt"),
        "checkpoint": os.path.join(
            ROOT, "transformer_graph_defect_residual_model.pt"
        ),
        "simulations_dir": os.path.join(ROOT, "SIMULATIONS"),
        "output_subdir": "point_transformer",
    },
    "planar_cgcnn": {
        "domain": "planar",
        "architecture": "cgcnn",
        "dataset": os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15.pt"),
        "checkpoint": os.path.join(ROOT, "cgcnn_planar_residual_c14c15_model.pt"),
        "initial_dir": os.path.join(ROOT, "Laves_Planar_Defects", "SIMULATIONS"),
        "relaxed_dir": os.path.join(ROOT, "Laves_Screen", "SIMULATIONS"),
        "output_subdir": "planar_cgcnn",
    },
    "planar_transformer": {
        "domain": "planar",
        "architecture": "transformer",
        "dataset": os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15.pt"),
        "checkpoint": os.path.join(
            ROOT, "transformer_graph_planar_residual_c14c15_model.pt"
        ),
        "initial_dir": os.path.join(ROOT, "Laves_Planar_Defects", "SIMULATIONS"),
        "relaxed_dir": os.path.join(ROOT, "Laves_Screen", "SIMULATIONS"),
        "output_subdir": "planar_transformer",
    },
}

GLOBAL_V2_JOBS: Dict[str, Dict[str, str]] = {
    "point_cgcnn": {
        **JOBS["point_cgcnn"],
        "dataset": os.path.join(ROOT, "adv_datasets", "cycle34_residual_k13_dataset.pt"),
        "checkpoint": os.path.join(ROOT, "cgcnn_defect_residual_global_v2_model.pt"),
    },
    "point_transformer": {
        **JOBS["point_transformer"],
        "dataset": os.path.join(ROOT, "adv_datasets", "cycle34_residual_k13_dataset.pt"),
        "checkpoint": os.path.join(
            ROOT, "transformer_graph_defect_residual_global_v2_model.pt"
        ),
    },
    "planar_cgcnn": {
        **JOBS["planar_cgcnn"],
        "checkpoint": os.path.join(
            ROOT, "cgcnn_planar_residual_global_v2_model.pt"
        ),
    },
    "planar_transformer": {
        **JOBS["planar_transformer"],
        "checkpoint": os.path.join(
            ROOT, "transformer_graph_planar_residual_global_v2_model.pt"
        ),
    },
}


def get_jobs(profile: str) -> Dict[str, Dict[str, str]]:
    if profile in {"default", "legacy"}:
        return JOBS
    if profile == "global_v2":
        return GLOBAL_V2_JOBS
    raise ValueError(f"Unknown job profile: {profile}")


@dataclass
class CrystalTiming:
    t_preprocess_s: float
    t_predict_s: float
    t_postprocess_s: float

    @property
    def t_total_s(self) -> float:
        return self.t_preprocess_s + self.t_predict_s + self.t_postprocess_s


@dataclass
class CrystalPETable:
    atom_id: np.ndarray
    pe_initial: np.ndarray
    pe_true: np.ndarray
    pe_pred: np.ndarray
    in_graph: np.ndarray


@dataclass
class DumpFrame:
    atom_ids: np.ndarray
    positions: np.ndarray
    pe: np.ndarray
    cell_matrix: np.ndarray
    pbc: Tuple[bool, bool, bool]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_nonempty_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _parse_box_bounds(
    lines: Sequence[str],
    bounds_header: str,
) -> Tuple[np.ndarray, Tuple[bool, bool, bool]]:
    """Parse LAMMPS ``ITEM: BOX BOUNDS`` into a 3x3 cell matrix + PBC flags."""
    tokens = bounds_header.split()
    pbc_tokens = tokens[-3:]
    pbc = tuple(t.lower() == "pp" for t in pbc_tokens)
    if len(pbc) != 3:
        pbc = (True, True, True)

    rows = [list(map(float, lines[i].split())) for i in range(3)]
    triclinic = "xy" in tokens

    if triclinic:
        xlo_bound, xhi_bound, xy = rows[0]
        ylo_bound, yhi_bound, xz = rows[1]
        zlo_bound, zhi_bound, yz = rows[2]
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
        zlo, zhi = zlo_bound, zhi_bound
        cell = np.array(
            [
                [xhi - xlo, xy, xz],
                [0.0, yhi - ylo, yz],
                [0.0, 0.0, zhi - zlo],
            ],
            dtype=np.float64,
        ).T
    else:
        xlo, xhi = rows[0][0], rows[0][1]
        ylo, yhi = rows[1][0], rows[1][1]
        zlo, zhi = rows[2][0], rows[2][1]
        cell = np.diag([xhi - xlo, yhi - ylo, zhi - zlo]).astype(np.float64)

    return cell, (bool(pbc[0]), bool(pbc[1]), bool(pbc[2]))


def read_lammps_dump(dump_path: str, pe_candidates: Sequence[str]) -> DumpFrame:
    """Read atom ids, positions, PE, and cell from a LAMMPS dump (no OVITO)."""
    with open(dump_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    i = 0
    n_atoms = None
    bounds_header = None
    bounds_lines: List[str] = []
    atoms_header = None
    atoms_start = None

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            n_atoms = int(lines[i + 1].split()[0])
            i += 2
            continue
        if line.startswith("ITEM: BOX BOUNDS"):
            bounds_header = line
            bounds_lines = [lines[i + 1], lines[i + 2], lines[i + 3]]
            i += 4
            continue
        if line.startswith("ITEM: ATOMS"):
            atoms_header = line.split()[2:]
            atoms_start = i + 1
            break
        i += 1

    if (
        n_atoms is None
        or bounds_header is None
        or atoms_header is None
        or atoms_start is None
    ):
        raise ValueError(f"Incomplete LAMMPS dump header in {dump_path}")

    col = {name: idx for idx, name in enumerate(atoms_header)}
    if "id" not in col:
        raise ValueError(f"Missing id column in {dump_path}")

    if all(k in col for k in ("xu", "yu", "zu")):
        x_key, y_key, z_key = "xu", "yu", "zu"
    elif all(k in col for k in ("x", "y", "z")):
        x_key, y_key, z_key = "x", "y", "z"
    else:
        raise ValueError(f"Missing position columns in {dump_path}")

    pe_key = next((c for c in pe_candidates if c in col), None)
    if pe_key is None:
        raise ValueError(
            f"Missing PE column in {dump_path}; have {atoms_header}, "
            f"tried {list(pe_candidates)}"
        )

    atom_ids = np.empty(n_atoms, dtype=np.int64)
    positions = np.empty((n_atoms, 3), dtype=np.float64)
    pe = np.empty(n_atoms, dtype=np.float64)
    for row_i in range(n_atoms):
        parts = lines[atoms_start + row_i].split()
        atom_ids[row_i] = int(float(parts[col["id"]]))
        positions[row_i, 0] = float(parts[col[x_key]])
        positions[row_i, 1] = float(parts[col[y_key]])
        positions[row_i, 2] = float(parts[col[z_key]])
        pe[row_i] = float(parts[col[pe_key]])

    cell_matrix, pbc = _parse_box_bounds(bounds_lines, bounds_header)
    return DumpFrame(
        atom_ids=atom_ids,
        positions=positions,
        pe=pe,
        cell_matrix=cell_matrix,
        pbc=pbc,
    )


def _load_dump_pe(
    dump_path: str,
    pe_candidates: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    frame = read_lammps_dump(dump_path, pe_candidates)
    return frame.atom_ids, frame.pe


def _pe_by_id(atom_ids: np.ndarray, pe: np.ndarray) -> Dict[int, float]:
    return {int(pid): float(val) for pid, val in zip(atom_ids, pe)}


def _shell_threshold(sorted_distances: np.ndarray, k_shells: int) -> float:
    shell_tol = max(float(sorted_distances[0]) * SHELL_TOL_REL, 1e-6)
    shell_distances = [float(sorted_distances[0])]
    for dist in sorted_distances[1:]:
        if abs(float(dist) - shell_distances[-1]) > shell_tol:
            shell_distances.append(float(dist))
    cutoff_idx = min(k_shells, len(shell_distances)) - 1
    return shell_distances[cutoff_idx]


def _point_subset_orig_indices(
    unrelaxed_dump: str,
    defect_id: int,
    cutoff_k: int = DEFECT_CUTOFF_K,
    cutoff_radius: float = DEFECT_CUTOFF_RADIUS,
    cutoff_mode: str = "shell",
) -> Tuple[np.ndarray, np.ndarray]:
    frame = read_lammps_dump(unrelaxed_dump, POINT_PE_CANDIDATES)
    particle_ids = frame.atom_ids
    id_to_index = {int(pid): idx for idx, pid in enumerate(particle_ids)}
    if int(defect_id) not in id_to_index:
        raise ValueError(f"Defect id {defect_id} not found in {unrelaxed_dump}")
    defect_index = id_to_index[int(defect_id)]

    positions = frame.positions
    cell_matrix = frame.cell_matrix
    pbc = frame.pbc
    try:
        inv_cell = np.linalg.inv(cell_matrix)
    except np.linalg.LinAlgError:
        inv_cell = np.linalg.pinv(cell_matrix)
    frac_positions = positions @ inv_cell

    all_dist = np.zeros(len(positions), dtype=float)
    for idx in range(len(positions)):
        if idx == defect_index:
            continue
        dfrac = frac_positions[idx] - frac_positions[defect_index]
        for dim in range(3):
            if pbc[dim]:
                dfrac[dim] -= np.round(dfrac[dim])
        all_dist[idx] = float(np.linalg.norm(dfrac @ cell_matrix))

    tol = 1e-6
    if cutoff_mode == "shell":
        non_self = np.array([d for d in all_dist if d > 0.0])
        cutoff_dist = (
            _shell_threshold(np.sort(non_self), cutoff_k) if len(non_self) else 0.0
        )
    elif cutoff_mode == "radius":
        cutoff_dist = float(cutoff_radius)
    else:
        raise ValueError(f"Unsupported cutoff_mode: {cutoff_mode}")

    subset = [idx for idx, dist in enumerate(all_dist) if dist <= cutoff_dist + tol]
    if defect_index not in subset:
        subset.append(defect_index)
    subset = sorted(set(subset))
    return particle_ids, np.asarray(subset, dtype=np.int64)


def _match_graph_nodes_by_pe(
    data: Data,
    atom_ids: np.ndarray,
    pe_initial: np.ndarray,
    pe_true: np.ndarray,
    atol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback: map graph nodes to full-cell rows via (PE_initial, PE_true)."""
    pe0 = data.x[:, 1].detach().cpu().numpy().reshape(-1)
    y = data.y.detach().cpu().numpy().reshape(-1)
    target_mode = None
    if isinstance(getattr(data, "meta", None), dict):
        target_mode = data.meta.get("target_mode")
    pe_t = y if target_mode == "absolute" else (pe0 + y)

    used = np.zeros(len(atom_ids), dtype=bool)
    graph_pos = np.empty(len(pe0), dtype=np.int64)
    for i, (a, b) in enumerate(zip(pe0, pe_t)):
        hits = np.where(
            (~used)
            & np.isclose(pe_initial, a, atol=atol, rtol=0.0)
            & np.isclose(pe_true, b, atol=atol, rtol=0.0)
        )[0]
        if len(hits) == 0:
            hits = np.where((~used) & np.isclose(pe_initial, a, atol=atol, rtol=0.0))[0]
        if len(hits) == 0:
            raise RuntimeError(
                f"Could not match graph node {i} (pe0={a}, pe_t={b}) to dump atoms"
            )
        j = int(hits[0])
        used[j] = True
        graph_pos[i] = j
    return atom_ids[graph_pos], graph_pos


def resolve_paired_paths(
    folder: str,
    initial_dir: str,
    relaxed_dir: str,
) -> Optional[Tuple[str, str, str]]:
    initial_folder = os.path.join(initial_dir, folder)
    relaxed_folder = os.path.join(relaxed_dir, folder)
    if not os.path.isdir(initial_folder) or not os.path.isdir(relaxed_folder):
        return None

    basefile = os.path.join(initial_folder, "basefile.data")
    if not _is_nonempty_file(basefile):
        basefile = os.path.join(relaxed_folder, "basefile.data")
    if not _is_nonempty_file(basefile):
        return None

    initial_dump = None
    for name in ("unrelaxed.dump", "initial.dump", "pre_relax.dump", "minimised.dump"):
        candidate = os.path.join(initial_folder, name)
        if _is_nonempty_file(candidate):
            initial_dump = candidate
            break
    if initial_dump is None:
        return None

    relaxed_dump = os.path.join(relaxed_folder, "minimised.dump")
    if not _is_nonempty_file(relaxed_dump):
        return None
    if os.path.normcase(os.path.abspath(initial_dump)) == os.path.normcase(
        os.path.abspath(relaxed_dump)
    ):
        return None
    return basefile, initial_dump, relaxed_dump


def point_config_key(data: Data) -> str:
    return (
        f"{int(data.defect_id)}-{int(data.from_type)}-"
        f"{int(data.to_type)}-{data.wyckoff}"
    )


def resolve_point_dump_paths(
    data: Data,
    simulations_dir: str,
) -> Tuple[str, str, str]:
    folder = str(data.folder)
    key = point_config_key(data)
    folder_path = os.path.join(simulations_dir, folder)
    unrelaxed_dump = os.path.join(folder_path, f"unrelaxed_{key}.dump")
    relaxed_dump = os.path.join(folder_path, f"relaxed_{key}.dump")
    data_path = os.path.join(folder_path, f"unrelaxed_{key}.data")
    for path, label in (
        (unrelaxed_dump, "unrelaxed dump"),
        (relaxed_dump, "relaxed dump"),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {label}: {path}")
    return unrelaxed_dump, relaxed_dump, data_path


def load_point_full_cell(
    data: Data,
    simulations_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unrelaxed_dump, relaxed_dump, _data_path = resolve_point_dump_paths(
        data, simulations_dir
    )
    atom_ids, pe_initial = _load_dump_pe(unrelaxed_dump, POINT_PE_CANDIDATES)
    relaxed_ids, relaxed_pe = _load_dump_pe(relaxed_dump, POINT_PE_CANDIDATES)
    relaxed_map = _pe_by_id(relaxed_ids, relaxed_pe)
    pe_true = np.array([relaxed_map[int(pid)] for pid in atom_ids], dtype=np.float64)

    if getattr(data, "particle_ids", None) is not None:
        graph_pids = np.asarray(data.particle_ids.cpu().numpy(), dtype=np.int64)
        id_to_pos = {int(pid): i for i, pid in enumerate(atom_ids)}
        graph_pos = np.array(
            [id_to_pos[int(pid)] for pid in graph_pids], dtype=np.int64
        )
        return atom_ids, pe_initial, pe_true, graph_pids, graph_pos

    cutoff_k = int(getattr(data, "cutoff_k", DEFECT_CUTOFF_K))
    cutoff_radius = float(getattr(data, "cutoff_radius", DEFECT_CUTOFF_RADIUS))
    cutoff_mode = str(getattr(data, "cutoff_mode", "shell"))
    try:
        _pids_full, orig_indices = _point_subset_orig_indices(
            unrelaxed_dump,
            defect_id=int(data.defect_id),
            cutoff_k=cutoff_k,
            cutoff_radius=cutoff_radius,
            cutoff_mode=cutoff_mode,
        )
        if len(orig_indices) != int(data.num_nodes):
            raise RuntimeError(
                f"Subset size mismatch: recomputed {len(orig_indices)} "
                f"vs graph {data.num_nodes}"
            )
        graph_pos = orig_indices
        graph_pids = atom_ids[graph_pos]
    except Exception:
        graph_pids, graph_pos = _match_graph_nodes_by_pe(
            data, atom_ids, pe_initial, pe_true
        )

    return atom_ids, pe_initial, pe_true, graph_pids, graph_pos


def load_planar_full_cell(
    data: Data,
    initial_dir: str,
    relaxed_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    folder = str(data.folder)
    resolved = resolve_paired_paths(folder, initial_dir, relaxed_dir)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not resolve paired planar paths for {folder!r}"
        )
    _basefile, initial_dump, relaxed_dump = resolved
    atom_ids, pe_initial = _load_dump_pe(initial_dump, PLANAR_PE_CANDIDATES)
    relaxed_ids, relaxed_pe = _load_dump_pe(relaxed_dump, PLANAR_PE_CANDIDATES)
    relaxed_map = _pe_by_id(relaxed_ids, relaxed_pe)
    pe_true = np.array([relaxed_map[int(pid)] for pid in atom_ids], dtype=np.float64)

    if getattr(data, "particle_ids", None) is not None:
        graph_pids = np.asarray(data.particle_ids.cpu().numpy(), dtype=np.int64)
    else:
        graph_pids = atom_ids.copy()
        if len(graph_pids) != int(data.num_nodes):
            raise RuntimeError(
                f"Planar atom count mismatch for {folder}: "
                f"dump {len(graph_pids)} vs graph {data.num_nodes}"
            )

    id_to_pos = {int(pid): i for i, pid in enumerate(atom_ids)}
    graph_pos = np.array([id_to_pos[int(pid)] for pid in graph_pids], dtype=np.int64)
    return atom_ids, pe_initial, pe_true, graph_pids, graph_pos


def restore_absolute_predictions(
    atom_ids: np.ndarray,
    pe_initial: np.ndarray,
    pe_true: np.ndarray,
    graph_pos: np.ndarray,
    delta_pred: np.ndarray,
) -> CrystalPETable:
    """Scatter residual predictions into a full-cell absolute PE table.

    Atoms not in the graph keep ``pe_pred = pe_initial`` (ΔPE = 0).
    """
    pe_pred = pe_initial.copy()
    in_graph = np.zeros(len(atom_ids), dtype=np.int64)
    delta = np.asarray(delta_pred, dtype=np.float64).reshape(-1)
    if len(delta) != len(graph_pos):
        raise ValueError(
            f"Prediction length {len(delta)} != graph map length {len(graph_pos)}"
        )
    pe_pred[graph_pos] = pe_initial[graph_pos] + delta
    in_graph[graph_pos] = 1
    return CrystalPETable(
        atom_id=atom_ids.astype(np.int64),
        pe_initial=pe_initial.astype(np.float64),
        pe_true=pe_true.astype(np.float64),
        pe_pred=pe_pred.astype(np.float64),
        in_graph=in_graph,
    )


def write_pe_csv(path: str, table: CrystalPETable, include_in_graph: bool) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["atom_id", "pe_initial", "pe_true", "pe_pred"]
    if include_in_graph:
        fieldnames.append("in_graph")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(table.atom_id)):
            row = {
                "atom_id": int(table.atom_id[i]),
                "pe_initial": float(table.pe_initial[i]),
                "pe_true": float(table.pe_true[i]),
                "pe_pred": float(table.pe_pred[i]),
            }
            if include_in_graph:
                row["in_graph"] = int(table.in_graph[i])
            writer.writerow(row)


def _cluster_timing_only(timing: Dict) -> Dict:
    """Keep only cluster inference wall-clock fields."""
    keys = (
        "t_preprocess_s",
        "t_predict_s",
        "t_postprocess_s",
        "t_total_s",
        "device",
    )
    return {k: timing[k] for k in keys if k in timing and timing[k] is not None}


def write_merged_timing_json(
    path: str,
    *,
    cgcnn_timing: Dict,
    xf_timing: Dict,
) -> None:
    """Write per-crystal timing JSON (cluster inference wall-clock only)."""
    entries: List[Tuple[str, object]] = [
        ("cgcnn_timing", _cluster_timing_only(cgcnn_timing)),
        ("transformer_timing", _cluster_timing_only(xf_timing)),
    ]

    lines: List[str] = ["{"]
    for i, (key, val) in enumerate(entries):
        comma = "," if i < len(entries) - 1 else ""
        if isinstance(val, dict):
            lines.append(f'  "{key}": {{')
            dict_items = list(val.items())
            for j, (sub_key, sub_val) in enumerate(dict_items):
                sub_comma = "," if j < len(dict_items) - 1 else ""
                lines.append(f'    "{sub_key}": {json.dumps(sub_val)}{sub_comma}')
            lines.append(f"  }}{comma}")
        else:
            lines.append(f'  "{key}": {json.dumps(val)}{comma}')
    lines.append("}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_merged_pe_csv(
    path: str,
    atom_ids: np.ndarray,
    pe_initial: np.ndarray,
    pe_true: np.ndarray,
    pe_pred_cgcnn: np.ndarray,
    pe_pred_transformer: np.ndarray,
    in_graph: np.ndarray,
    *,
    domain: str,
    config: str,
    include_in_graph: bool,
    split: str = "train",
    mae_cgcnn_abs_eV: float,
    mae_transformer_abs_eV: float,
    cgcnn_mae_residual_eV: Optional[float],
    transformer_mae_residual_eV: Optional[float],
) -> Dict[str, float]:
    """Write one crystal CSV with error metrics in the header (no runtimes).

    Wall-clock timings live in the companion ``*_timing.json`` file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pe_true_total = float(np.sum(pe_true))
    pe_initial_total = float(np.sum(pe_initial))
    pe_cgcnn_total = float(np.sum(pe_pred_cgcnn))
    pe_xf_total = float(np.sum(pe_pred_transformer))
    totals = {
        "pe_initial_total_eV": pe_initial_total,
        "pe_true_total_eV": pe_true_total,
        "pe_pred_cgcnn_total_eV": pe_cgcnn_total,
        "pe_pred_transformer_total_eV": pe_xf_total,
        "pe_error_cgcnn_total_eV": pe_cgcnn_total - pe_true_total,
        "pe_error_transformer_total_eV": pe_xf_total - pe_true_total,
    }
    errors = {
        "mae_cgcnn_abs_eV": float(mae_cgcnn_abs_eV),
        "mae_transformer_abs_eV": float(mae_transformer_abs_eV),
        "cgcnn_mae_residual_eV": float(cgcnn_mae_residual_eV)
        if cgcnn_mae_residual_eV is not None
        else None,
        "transformer_mae_residual_eV": float(transformer_mae_residual_eV)
        if transformer_mae_residual_eV is not None
        else None,
    }
    if split not in {"train", "validation"}:
        raise ValueError(f"split must be 'train' or 'validation', got {split!r}")

    fieldnames = [
        "atom_id",
        "pe_initial",
        "pe_true",
        "pe_pred_cgcnn",
        "pe_pred_transformer",
    ]
    if include_in_graph:
        fieldnames.append("in_graph")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# domain={domain}\n")
        fh.write(f"# config={config}\n")
        fh.write(f"# split={split}\n")
        fh.write("\n")
        fh.write("# --- total system PE (sum over all atoms, eV) ---\n")
        for key, val in totals.items():
            fh.write(f"# {key}={val}\n")
        fh.write("\n")
        fh.write("# --- prediction errors (eV) ---\n")
        for key, val in errors.items():
            if val is not None:
                fh.write(f"# {key}={val}\n")
        fh.write("\n")
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(atom_ids)):
            row = {
                "atom_id": int(atom_ids[i]),
                "pe_initial": float(pe_initial[i]),
                "pe_true": float(pe_true[i]),
                "pe_pred_cgcnn": float(pe_pred_cgcnn[i]),
                "pe_pred_transformer": float(pe_pred_transformer[i]),
            }
            if include_in_graph:
                row["in_graph"] = int(in_graph[i])
            writer.writerow(row)
    return {**totals, **{k: v for k, v in errors.items() if v is not None}}


def write_variables_readme(predictions_root: str) -> str:
    """Write a single variable glossary for point + planar deliveries."""
    text = """Variable guide for defect prediction outputs
=============================================

Deliverables live under predictions/point/ and predictions/planar/. Each
crystal has a CSV (comment header + one row per atom) and a companion
``*_timing.json`` with wall-clock seconds only.

Point vs planar (only differences)
----------------------------------
File layout:
  predictions/point/<folder>/<config>.csv     e.g. .../C15_Ag-Be/1-1-1-8a.csv
  predictions/planar/<folder>/pe_table.csv    config equals folder name

Data columns:
  Point defects include in_graph (see below). Planar graphs cover the full
  crystal — every atom is predicted; there is no in_graph column.

Everything below applies to both domains unless noted.

Header (identity / split)
-------------------------
domain, config
  Crystal identifiers. The parent directory mirrors the crystal folder name.

split
  train or validation — which residual-training split this crystal belonged to.
  Validation crystals were not used to update weights; prefer them when judging
  generalization. See predictions/SPLIT_INFO.txt.

Timings (*_timing.json)
-----------------------
Each CSV has a companion ``<stem>_timing.json`` with wall-clock seconds only
(no error metrics). Three blocks:

Local restore (dump I/O + CSV write on this machine):
  t_preprocess_s
    Read LAMMPS dumps and map graph predictions onto the full cell.
  t_postprocess_s
    Write the CSV (and this timing file).
  t_total_s
    t_preprocess_s + t_postprocess_s

cgcnn_timing — cluster GPU inference for this crystal (batch size 1):
  t_preprocess_s
    Move graph batch to device.
  t_predict_s
    CGCNN forward pass (residual ΔPE).
  t_postprocess_s
    Save on-graph predictions to disk.
  t_total_s
    Sum of the three cluster phases above.
  device
    Hardware used for timed inference (usually cuda).

transformer_timing
  Same fields as cgcnn_timing, for the graph transformer on the cluster.

Header (total system PE) — in the CSV only
------------------------------------------
pe_initial_total_eV
  Sum of pe_initial over all atoms = total unrelaxed / initial system PE (eV).

pe_true_total_eV
  Sum of pe_true over all atoms = ground-truth total relaxed system PE (eV).

pe_pred_cgcnn_total_eV
  Sum of pe_pred_cgcnn over all atoms = CGCNN predicted total system PE (eV).

pe_pred_transformer_total_eV
  Sum of pe_pred_transformer over all atoms = transformer predicted total PE (eV).

pe_error_cgcnn_total_eV
  pe_pred_cgcnn_total_eV − pe_true_total_eV

pe_error_transformer_total_eV
  pe_pred_transformer_total_eV − pe_true_total_eV

Header (prediction errors) — in the CSV only
-------------------------------------------
mae_cgcnn_abs_eV
  Mean |pe_pred_cgcnn − pe_true| over all atoms (full cell).

mae_transformer_abs_eV
  Mean |pe_pred_transformer − pe_true| over all atoms (full cell).

cgcnn_mae_residual_eV
  Mean |ΔPE_pred − ΔPE_true| on graph atoms (CGCNN, cluster inference).

transformer_mae_residual_eV
  Mean |ΔPE_pred − ΔPE_true| on graph atoms (transformer, cluster inference).

Data columns
------------
atom_id
  LAMMPS atom id.

pe_initial
  Per-atom potential energy before relaxation (eV), from the initial/unrelaxed dump.

pe_true
  Per-atom ground-truth potential energy after relaxation (eV).

pe_pred_cgcnn
  Absolute PE from the residual CGCNN: pe_initial + ΔPE_pred (eV).

pe_pred_transformer
  Absolute PE from the residual graph transformer: pe_initial + ΔPE_pred (eV).

in_graph  (point defects only)
  1 if the atom was part of the defect subgraph used by the model;
  0 otherwise. Atoms with in_graph=0 keep pe_pred = pe_initial (ΔPE=0).

Units are eV throughout. Absolute predictions come from residual models trained
on ΔPE = pe_true − pe_initial.
"""
    os.makedirs(predictions_root, exist_ok=True)
    path = os.path.join(predictions_root, "VARIABLES.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    for legacy in ("VARIABLES_point.txt", "VARIABLES_planar.txt"):
        legacy_path = os.path.join(predictions_root, legacy)
        if os.path.isfile(legacy_path):
            os.remove(legacy_path)
    return path


def write_split_info(
    predictions_root: str,
    *,
    seed: int,
    val_fraction: float,
    point_n_train: int,
    point_n_val: int,
    planar_n_train: int,
    planar_n_val: int,
    split_json: Optional[str] = None,
) -> str:
    """Document how train/validation flags were assigned."""
    split_source = (
        f"Stored explicitly in {split_json} (train/val index lists)."
        if split_json
        else (
            "Recomputed locally with train_single.within_group_train_val_indices "
            f"(seed={seed}, val_fraction={val_fraction})."
        )
    )
    text = f"""Train / validation split flags
===============================

Each crystal CSV has ``# split=train`` or ``# split=validation``.

How this was determined
-----------------------
{split_source}

Recorded sizes:
  point:  n_train={point_n_train}, n_val={point_n_val}
  planar: n_train={planar_n_train}, n_val={planar_n_val}

What "validation" means here
----------------------------
Validation crystals were held out of the weight updates and used only for
model selection (best-by-val R_tot median for global-v2 delivery models).
They are the more honest estimate of error than training crystals.
There was no separate test set.

Caveats
-------
- Inference was still run on the full dataset (train + validation) for delivery.
- Prefer metrics aggregated over split=validation when reporting generalization.
- CGCNN and transformer share the same split indices.
"""
    os.makedirs(predictions_root, exist_ok=True)
    path = os.path.join(predictions_root, "SPLIT_INFO.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def graph_pred_paths(crystal_dir: str, stem: str) -> List[str]:
    """Candidate paths for on-graph residual predictions (npz preferred)."""
    return [
        os.path.join(crystal_dir, f"{stem}_graph_pred.npz"),
        os.path.join(crystal_dir, f"{stem}_graph_pred.pt"),
    ]


def save_graph_pred(path_npz: str, payload: Dict) -> str:
    """Save graph-level ΔPE as ``.npz`` (not caught by ``*.pt`` gitignore)."""
    arrays = {
        "delta_pred": np.asarray(payload["delta_pred"], dtype=np.float32).reshape(-1, 1),
        "pe_initial_graph": np.asarray(
            payload["pe_initial_graph"], dtype=np.float32
        ).reshape(-1, 1),
        "y_true_residual": np.asarray(
            payload["y_true_residual"], dtype=np.float32
        ).reshape(-1, 1),
    }
    if payload.get("particle_ids") is not None:
        arrays["particle_ids"] = np.asarray(
            payload["particle_ids"], dtype=np.int64
        ).reshape(-1)
    meta = {
        "folder": str(payload.get("folder", "")),
        "config": str(payload.get("config", "")),
        "domain": str(payload.get("domain", "")),
        "architecture": str(payload.get("architecture", "")),
        "n_atoms_graph": int(payload.get("n_atoms_graph", arrays["delta_pred"].shape[0])),
        "identity_pred": bool(payload.get("identity_pred", False)),
    }
    np.savez_compressed(path_npz, meta_json=json.dumps(meta), **arrays)
    return path_npz


def load_graph_pred(crystal_dir: str, stem: str) -> Dict:
    """Load ``*_graph_pred.npz`` or legacy ``*_graph_pred.pt``."""
    last_err: Optional[Exception] = None
    for path in graph_pred_paths(crystal_dir, stem):
        if not os.path.isfile(path):
            continue
        try:
            if path.endswith(".npz"):
                raw = np.load(path, allow_pickle=False)
                meta = {}
                if "meta_json" in raw.files:
                    meta_raw = raw["meta_json"]
                    meta = json.loads(
                        meta_raw.item() if getattr(meta_raw, "shape", ()) == () else str(meta_raw)
                    )
                out = {
                    "delta_pred": np.asarray(raw["delta_pred"], dtype=np.float64).reshape(
                        -1
                    ),
                    "pe_initial_graph": np.asarray(
                        raw["pe_initial_graph"], dtype=np.float64
                    ).reshape(-1, 1),
                    "y_true_residual": np.asarray(
                        raw["y_true_residual"], dtype=np.float64
                    ).reshape(-1, 1),
                    "particle_ids": (
                        np.asarray(raw["particle_ids"], dtype=np.int64).reshape(-1)
                        if "particle_ids" in raw.files
                        else None
                    ),
                    **meta,
                    "pred_path": path,
                }
                return out
            payload = torch.load(path, weights_only=False)
            out = {
                "delta_pred": np.asarray(
                    payload["delta_pred"], dtype=np.float64
                ).reshape(-1),
                "pe_initial_graph": np.asarray(
                    payload["pe_initial_graph"], dtype=np.float64
                ).reshape(-1, 1),
                "y_true_residual": np.asarray(
                    payload["y_true_residual"], dtype=np.float64
                ).reshape(-1, 1),
                "particle_ids": (
                    np.asarray(payload["particle_ids"], dtype=np.int64).reshape(-1)
                    if payload.get("particle_ids") is not None
                    else None
                ),
                "folder": payload.get("folder"),
                "config": payload.get("config"),
                "domain": payload.get("domain"),
                "architecture": payload.get("architecture"),
                "n_atoms_graph": payload.get("n_atoms_graph"),
                "identity_pred": payload.get("identity_pred", False),
                "pred_path": path,
            }
            return out
        except Exception as err:  # noqa: BLE001 - try next candidate
            last_err = err
            continue
    tried = ", ".join(graph_pred_paths(crystal_dir, stem))
    raise FileNotFoundError(
        f"Missing graph pred under {crystal_dir!r} (tried {tried})"
        + (f"; last error: {last_err}" if last_err else "")
    )


def load_timing_json(crystal_dir: str, stem: str) -> Dict:
    path = os.path.join(crystal_dir, f"{stem}_timing.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_pe_pt(path: str, table: CrystalPETable) -> None:
    """Store the same PE columns as float tensors (dataset-style)."""
    payload = {
        "atom_id": torch.tensor(table.atom_id, dtype=torch.long),
        "pe_initial": torch.tensor(table.pe_initial, dtype=torch.float).view(-1, 1),
        "pe_true": torch.tensor(table.pe_true, dtype=torch.float).view(-1, 1),
        "pe_pred": torch.tensor(table.pe_pred, dtype=torch.float).view(-1, 1),
        "in_graph": torch.tensor(table.in_graph, dtype=torch.long),
    }
    torch.save(payload, path)


def load_residual_model(
    checkpoint_path: str,
    dataset: List[Data],
    architecture: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    # Training configs often mix optimiser knobs into the same dict.
    model_keys = {
        "hidden_dim",
        "num_layers",
        "dropout",
        "use_batch_norm",
        "activation",
        "bidirectional",
        "num_heads",
        "attention_dropout",
        "out_dim",
    }
    model_cfg = {k: v for k, v in config.items() if k in model_keys}

    if architecture == "cgcnn":
        model = build_gated_model_from_dataset(
            dataset,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            use_batch_norm=bool(model_cfg.get("use_batch_norm", False)),
            activation=str(model_cfg.get("activation", "silu")),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
            out_dim=int(model_cfg.get("out_dim", 1)),
        )
    elif architecture == "transformer":
        model = build_graph_transformer_from_dataset(
            dataset,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            attention_dropout=float(model_cfg.get("attention_dropout", 0.1)),
            activation=str(model_cfg.get("activation", "gelu")),
            out_dim=int(model_cfg.get("out_dim", 1)),
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")

    state = ckpt["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    target_mean = torch.tensor(float(ckpt["target_mean"]), device=device)
    target_std = torch.tensor(float(ckpt["target_std"]), device=device)
    return model, target_mean, target_std


def crystal_path_info(data: Data, domain: str) -> Tuple[str, str, str, str]:
    """Return ``(folder, config_key, rel_dir, stem)`` for mirrored outputs."""
    folder = str(data.folder)
    if domain == "point":
        config_key = point_config_key(data)
        return folder, config_key, folder, config_key
    return folder, folder, folder, "pe_table"


def predict_residual_delta(
    model: Optional[torch.nn.Module],
    data: Data,
    device: torch.device,
    target_mean: Optional[torch.Tensor],
    target_std: Optional[torch.Tensor],
    identity_pred: bool,
) -> Tuple[np.ndarray, float, float]:
    """Return residual ΔPE, preprocess seconds (to device), predict seconds."""
    if identity_pred:
        t0 = time.perf_counter()
        delta = data.y.detach().cpu().numpy().reshape(-1)
        dt = time.perf_counter() - t0
        return delta, 0.0, dt

    assert model is not None and target_mean is not None and target_std is not None
    loader = DataLoader([data], batch_size=1, shuffle=False)
    t_pre0 = time.perf_counter()
    batch = next(iter(loader)).to(device)
    _sync(device)
    t_preprocess = time.perf_counter() - t_pre0

    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        pred_norm = model(batch)
        pred = pred_norm * target_std + target_mean
        _sync(device)
        t_predict = time.perf_counter() - t0
    return pred.detach().cpu().numpy().reshape(-1), t_preprocess, t_predict


def infer_one_crystal(
    data: Data,
    job: Dict[str, str],
    out_root: str,
    device: torch.device,
    model: Optional[torch.nn.Module],
    target_mean: Optional[torch.Tensor],
    target_std: Optional[torch.Tensor],
    identity_pred: bool,
) -> Dict:
    """Run model inference only (no dumps). Saves ΔPE + timing for later restore."""
    domain = job["domain"]
    folder, config_key, rel_dir, stem = crystal_path_info(data, domain)
    crystal_dir = os.path.join(out_root, rel_dir)
    os.makedirs(crystal_dir, exist_ok=True)

    delta, t_preprocess, t_predict = predict_residual_delta(
        model=model,
        data=data,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        identity_pred=identity_pred,
    )

    t_post0 = time.perf_counter()
    pe_initial_graph = data.x[:, 1:2].detach().cpu().numpy()
    particle_ids = None
    if getattr(data, "particle_ids", None) is not None:
        particle_ids = data.particle_ids.detach().cpu().numpy().astype(np.int64)
    pred_path = os.path.join(crystal_dir, f"{stem}_graph_pred.npz")
    save_graph_pred(
        pred_path,
        {
            "delta_pred": delta,
            "pe_initial_graph": pe_initial_graph,
            "y_true_residual": data.y.detach().cpu().numpy().reshape(-1, 1),
            "particle_ids": particle_ids,
            "folder": folder,
            "config": config_key,
            "domain": domain,
            "architecture": job["architecture"],
            "n_atoms_graph": int(data.num_nodes),
            "identity_pred": bool(identity_pred),
        },
    )
    t_postprocess = time.perf_counter() - t_post0

    # On-graph residual MAE vs dataset labels (no dumps needed).
    y_true = data.y.detach().cpu().numpy().reshape(-1)
    mae_resid = float(np.mean(np.abs(delta - y_true)))

    timing = CrystalTiming(
        t_preprocess_s=t_preprocess,
        t_predict_s=t_predict,
        t_postprocess_s=t_postprocess,
    )
    meta = {
        "folder": folder,
        "config": config_key,
        "domain": domain,
        "architecture": job["architecture"],
        "mode": "inference",
        "n_atoms_graph": int(data.num_nodes),
        "identity_pred": bool(identity_pred),
        "pred_path": pred_path,
        "mae_residual_eV": mae_resid,
        **asdict(timing),
        "t_total_s": timing.t_total_s,
        "device": str(device),
    }
    with open(
        os.path.join(crystal_dir, f"{stem}_timing.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(meta, fh, indent=2)
    return meta


def restore_one_crystal(
    data: Data,
    job: Dict[str, str],
    out_root: str,
    inference_root: str,
    write_pt: bool,
) -> Dict:
    """Restore full-crystal PE tables from a prior ``--mode inference`` run."""
    domain = job["domain"]
    folder, config_key, rel_dir, stem = crystal_path_info(data, domain)
    infer_dir = os.path.join(inference_root, job["output_subdir"], rel_dir)
    payload = load_graph_pred(infer_dir, stem)
    timing_meta = load_timing_json(infer_dir, stem)
    delta = np.asarray(payload["delta_pred"], dtype=np.float64).reshape(-1)
    pred_path = str(payload.get("pred_path", ""))

    t_pre0 = time.perf_counter()
    if domain == "point":
        atom_ids, pe_initial, pe_true, _gpids, graph_pos = load_point_full_cell(
            data, job["simulations_dir"]
        )
    else:
        atom_ids, pe_initial, pe_true, _gpids, graph_pos = load_planar_full_cell(
            data, job["initial_dir"], job["relaxed_dir"]
        )
    # Prefer stored particle_ids map when available.
    if payload.get("particle_ids") is not None:
        graph_pids = np.asarray(payload["particle_ids"], dtype=np.int64).reshape(-1)
        id_to_pos = {int(pid): i for i, pid in enumerate(atom_ids)}
        graph_pos = np.array(
            [id_to_pos[int(pid)] for pid in graph_pids], dtype=np.int64
        )
    t_preprocess = time.perf_counter() - t_pre0

    t_post0 = time.perf_counter()
    table = restore_absolute_predictions(
        atom_ids=atom_ids,
        pe_initial=pe_initial,
        pe_true=pe_true,
        graph_pos=graph_pos,
        delta_pred=delta,
    )
    crystal_dir = os.path.join(out_root, rel_dir)
    csv_path = os.path.join(crystal_dir, f"{stem}.csv")
    write_pe_csv(csv_path, table, include_in_graph=(domain == "point"))
    pt_path = None
    if write_pt:
        pt_path = os.path.join(crystal_dir, f"{stem}.pt")
        write_pe_pt(pt_path, table)
    t_postprocess = time.perf_counter() - t_post0

    meta = {
        "folder": folder,
        "config": config_key,
        "domain": domain,
        "architecture": job["architecture"],
        "mode": "restore",
        "n_atoms_full": int(len(atom_ids)),
        "n_atoms_graph": int(data.num_nodes),
        "n_atoms_in_graph_flag": int(table.in_graph.sum()),
        "pred_path": pred_path,
        "csv_path": csv_path,
        "pt_path": pt_path,
        "t_preprocess_s": t_preprocess,
        "t_predict_s": float(timing_meta.get("t_predict_s", 0.0)),
        "t_postprocess_s": t_postprocess,
        "t_total_s": t_preprocess + t_postprocess,
        "cluster_t_predict_s": timing_meta.get("t_predict_s"),
        "cluster_t_preprocess_s": timing_meta.get("t_preprocess_s"),
        "cluster_device": timing_meta.get("device"),
        "mae_abs_eV": float(np.mean(np.abs(table.pe_pred - table.pe_true))),
        "mae_in_graph_abs_eV": float(
            np.mean(
                np.abs(
                    table.pe_pred[table.in_graph == 1]
                    - table.pe_true[table.in_graph == 1]
                )
            )
        )
        if table.in_graph.any()
        else None,
    }
    with open(
        os.path.join(crystal_dir, f"{stem}_timing.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(meta, fh, indent=2)
    return meta


def restore_merged_one_crystal(
    data: Data,
    domain: str,
    out_root: str,
    inference_root: str,
    split: str,
    jobs: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict:
    """Restore one crystal using both CGCNN and Transformer inference outputs."""
    job_map = jobs or JOBS
    cgcnn_job = job_map[f"{domain}_cgcnn"]
    xf_job = job_map[f"{domain}_transformer"]
    folder, config_key, rel_dir, stem = crystal_path_info(data, domain)

    cgcnn_dir = os.path.join(inference_root, cgcnn_job["output_subdir"], rel_dir)
    xf_dir = os.path.join(inference_root, xf_job["output_subdir"], rel_dir)
    cgcnn_payload = load_graph_pred(cgcnn_dir, stem)
    xf_payload = load_graph_pred(xf_dir, stem)
    cgcnn_timing = load_timing_json(cgcnn_dir, stem)
    xf_timing = load_timing_json(xf_dir, stem)

    delta_c = np.asarray(cgcnn_payload["delta_pred"], dtype=np.float64).reshape(-1)
    delta_t = np.asarray(xf_payload["delta_pred"], dtype=np.float64).reshape(-1)
    if len(delta_c) != len(delta_t):
        raise ValueError(
            f"ΔPE length mismatch for {folder}/{config_key}: "
            f"cgcnn={len(delta_c)} transformer={len(delta_t)}"
        )

    t_pre0 = time.perf_counter()
    if domain == "point":
        atom_ids, pe_initial, pe_true, _gpids, graph_pos = load_point_full_cell(
            data, cgcnn_job["simulations_dir"]
        )
    else:
        atom_ids, pe_initial, pe_true, _gpids, graph_pos = load_planar_full_cell(
            data, cgcnn_job["initial_dir"], cgcnn_job["relaxed_dir"]
        )
    pids = cgcnn_payload.get("particle_ids")
    if pids is None:
        pids = xf_payload.get("particle_ids")
    if pids is not None:
        graph_pids = np.asarray(pids, dtype=np.int64).reshape(-1)
        id_to_pos = {int(pid): i for i, pid in enumerate(atom_ids)}
        graph_pos = np.array(
            [id_to_pos[int(pid)] for pid in graph_pids], dtype=np.int64
        )
    t_preprocess = time.perf_counter() - t_pre0

    t_post0 = time.perf_counter()
    table_c = restore_absolute_predictions(
        atom_ids, pe_initial, pe_true, graph_pos, delta_c
    )
    table_t = restore_absolute_predictions(
        atom_ids, pe_initial, pe_true, graph_pos, delta_t
    )
    crystal_dir = os.path.join(out_root, rel_dir)
    csv_path = os.path.join(crystal_dir, f"{stem}.csv")
    mae_cgcnn = float(np.mean(np.abs(table_c.pe_pred - table_c.pe_true)))
    mae_xf = float(np.mean(np.abs(table_t.pe_pred - table_t.pe_true)))
    totals = write_merged_pe_csv(
        csv_path,
        atom_ids=atom_ids,
        pe_initial=pe_initial,
        pe_true=pe_true,
        pe_pred_cgcnn=table_c.pe_pred,
        pe_pred_transformer=table_t.pe_pred,
        in_graph=table_c.in_graph,
        domain=domain,
        config=config_key,
        include_in_graph=(domain == "point"),
        split=split,
        mae_cgcnn_abs_eV=mae_cgcnn,
        mae_transformer_abs_eV=mae_xf,
        cgcnn_mae_residual_eV=cgcnn_timing.get("mae_residual_eV"),
        transformer_mae_residual_eV=xf_timing.get("mae_residual_eV"),
    )
    t_postprocess = time.perf_counter() - t_post0

    meta = {
        "folder": folder,
        "config": config_key,
        "domain": domain,
        "split": split,
        "n_atoms_full": int(len(atom_ids)),
        "n_atoms_graph": int(data.num_nodes),
        "n_atoms_in_graph_flag": int(table_c.in_graph.sum()),
        "csv_path": csv_path,
        "cgcnn_pred_path": cgcnn_payload.get("pred_path"),
        "transformer_pred_path": xf_payload.get("pred_path"),
        "t_preprocess_s": t_preprocess,
        "t_postprocess_s": t_postprocess,
        "t_predict_s": float(cgcnn_timing.get("t_predict_s", 0.0))
        + float(xf_timing.get("t_predict_s", 0.0)),
        "t_total_s": t_preprocess + t_postprocess,
        "cgcnn_t_predict_s": cgcnn_timing.get("t_predict_s"),
        "transformer_t_predict_s": xf_timing.get("t_predict_s"),
        "mae_cgcnn_abs_eV": mae_cgcnn,
        "mae_transformer_abs_eV": mae_xf,
        "cgcnn_mae_residual_eV": cgcnn_timing.get("mae_residual_eV"),
        "transformer_mae_residual_eV": xf_timing.get("mae_residual_eV"),
        **totals,
    }
    write_merged_timing_json(
        os.path.join(crystal_dir, f"{stem}_timing.json"),
        cgcnn_timing=cgcnn_timing,
        xf_timing=xf_timing,
    )
    return meta


def run_merged_restore(
    domain: str,
    output_root: str,
    inference_root: str,
    limit: Optional[int],
    *,
    split_json: Optional[str] = None,
    jobs: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict:
    if domain not in {"point", "planar"}:
        raise SystemExit(f"Unknown domain {domain!r}")
    job_map = jobs or JOBS
    job = job_map[f"{domain}_cgcnn"]
    dataset_path = job["dataset"]
    if not os.path.isfile(dataset_path):
        raise SystemExit(f"Dataset not found: {dataset_path}")

    print(f"[{domain}] mode=restore-merged")
    print(f"[{domain}] dataset={dataset_path}")
    print(f"[{domain}] inference_root={inference_root}")

    dataset: List[Data] = torch.load(dataset_path, weights_only=False)
    if split_json and os.path.isfile(split_json):
        from delivery_global_v2 import load_delivery_split, val_index_set

        split_payload = load_delivery_split(split_json)
        val_set = val_index_set(split_payload, domain)
        train_n = int(split_payload[domain]["n_train"])
        val_n = int(split_payload[domain]["n_val"])
        print(
            f"[{domain}] split from {split_json}: train={train_n} val={val_n}",
            flush=True,
        )
        train_idx = split_payload[domain]["train_indices"]
        val_idx = split_payload[domain]["val_indices"]
    else:
        train_idx, val_idx = within_group_train_val_indices(
            dataset,
            seed=DELIVERY_SPLIT_SEED,
            val_fraction=DELIVERY_VAL_FRACTION,
        )
        val_set = set(int(i) for i in val_idx.tolist())
        print(
            f"[{domain}] reconstructed split seed={DELIVERY_SPLIT_SEED} "
            f"val_fraction={DELIVERY_VAL_FRACTION}: "
            f"train={len(train_idx)} val={len(val_idx)}"
        )

    if limit is not None:
        dataset = dataset[: max(0, int(limit))]
    print(f"[{domain}] crystals={len(dataset)}")

    out_root = os.path.join(output_root, domain)
    os.makedirs(out_root, exist_ok=True)
    variables_readme = os.path.join(output_root, "VARIABLES.txt")

    summaries: List[Dict] = []
    for i, data in enumerate(dataset):
        split = "validation" if i in val_set else "train"
        try:
            meta = restore_merged_one_crystal(
                data=data,
                domain=domain,
                out_root=out_root,
                inference_root=inference_root,
                split=split,
                jobs=jobs,
            )
        except Exception as err:
            print(
                f"  [{i + 1}/{len(dataset)}] FAIL "
                f"{getattr(data, 'folder', '?')}: {err}"
            )
            summaries.append(
                {
                    "folder": str(getattr(data, "folder", "")),
                    "error": str(err),
                    "split": split,
                }
            )
            continue
        summaries.append(meta)
        if (i + 1) % 25 == 0 or i == 0 or i + 1 == len(dataset):
            print(
                f"  [{i + 1}/{len(dataset)}] {meta['folder']}/{meta['config']} "
                f"split={meta['split']} "
                f"MAE_c={meta['mae_cgcnn_abs_eV']:.4e} "
                f"MAE_t={meta['mae_transformer_abs_eV']:.4e} "
                f"t_restore={meta['t_total_s'] * 1000:.1f} ms"
            )

    ok = [s for s in summaries if "error" not in s]
    ok_val = [s for s in ok if s.get("split") == "validation"]
    ok_train = [s for s in ok if s.get("split") == "train"]
    summary_path = os.path.join(out_root, "restore_merged_summary.json")

    def _mean_key(rows: List[Dict], key: str) -> Optional[float]:
        vals = [s[key] for s in rows if s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "domain": domain,
                "mode": "restore-merged",
                "dataset": dataset_path,
                "inference_root": inference_root,
                "split_seed": DELIVERY_SPLIT_SEED,
                "val_fraction": DELIVERY_VAL_FRACTION,
                "split_json": split_json,
                "n_train_split": int(len(train_idx) if isinstance(train_idx, list) else len(train_idx)),
                "n_val_split": int(len(val_idx) if isinstance(val_idx, list) else len(val_idx)),
                "n_requested": len(dataset),
                "n_ok": len(ok),
                "n_ok_train": len(ok_train),
                "n_ok_validation": len(ok_val),
                "n_failed": len(summaries) - len(ok),
                "mean_mae_cgcnn_abs_eV": _mean_key(ok, "mae_cgcnn_abs_eV"),
                "mean_mae_transformer_abs_eV": _mean_key(
                    ok, "mae_transformer_abs_eV"
                ),
                "mean_mae_cgcnn_abs_eV_validation": _mean_key(
                    ok_val, "mae_cgcnn_abs_eV"
                ),
                "mean_mae_transformer_abs_eV_validation": _mean_key(
                    ok_val, "mae_transformer_abs_eV"
                ),
                "mean_abs_pe_error_cgcnn_total_eV": float(
                    np.mean([abs(s["pe_error_cgcnn_total_eV"]) for s in ok])
                )
                if ok
                else None,
                "mean_abs_pe_error_transformer_total_eV": float(
                    np.mean([abs(s["pe_error_transformer_total_eV"]) for s in ok])
                )
                if ok
                else None,
                "mean_abs_pe_error_cgcnn_total_eV_validation": float(
                    np.mean([abs(s["pe_error_cgcnn_total_eV"]) for s in ok_val])
                )
                if ok_val
                else None,
                "mean_abs_pe_error_transformer_total_eV_validation": float(
                    np.mean(
                        [abs(s["pe_error_transformer_total_eV"]) for s in ok_val]
                    )
                )
                if ok_val
                else None,
                "mean_cgcnn_t_predict_s": _mean_key(ok, "cgcnn_t_predict_s"),
                "mean_transformer_t_predict_s": _mean_key(
                    ok, "transformer_t_predict_s"
                ),
                "variables_readme": variables_readme,
                "crystals": summaries,
            },
            fh,
            indent=2,
        )
    print(f"[{domain}] wrote {len(ok)} crystals -> {out_root}")
    print(f"[{domain}] summary -> {summary_path}")
    return {
        "n_train": int(len(train_idx) if isinstance(train_idx, list) else len(train_idx)),
        "n_val": int(len(val_idx) if isinstance(val_idx, list) else len(val_idx)),
    }


def export_one_crystal(
    data: Data,
    job: Dict[str, str],
    out_root: str,
    device: torch.device,
    model: Optional[torch.nn.Module],
    target_mean: Optional[torch.Tensor],
    target_std: Optional[torch.Tensor],
    identity_pred: bool,
    write_pt: bool,
) -> Dict:
    domain = job["domain"]
    include_in_graph = domain == "point"
    folder, config_key, rel_dir, stem = crystal_path_info(data, domain)

    t_pre0 = time.perf_counter()
    if domain == "point":
        atom_ids, pe_initial, pe_true, _gpids, graph_pos = load_point_full_cell(
            data, job["simulations_dir"]
        )
    else:
        atom_ids, pe_initial, pe_true, _gpids, graph_pos = load_planar_full_cell(
            data, job["initial_dir"], job["relaxed_dir"]
        )
    t_dump = time.perf_counter() - t_pre0

    delta, t_to_device, t_predict = predict_residual_delta(
        model=model,
        data=data,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        identity_pred=identity_pred,
    )
    t_preprocess = t_dump + t_to_device

    t_post0 = time.perf_counter()
    table = restore_absolute_predictions(
        atom_ids=atom_ids,
        pe_initial=pe_initial,
        pe_true=pe_true,
        graph_pos=graph_pos,
        delta_pred=delta,
    )
    crystal_dir = os.path.join(out_root, rel_dir)
    csv_path = os.path.join(crystal_dir, f"{stem}.csv")
    write_pe_csv(csv_path, table, include_in_graph=include_in_graph)
    pt_path = None
    if write_pt:
        pt_path = os.path.join(crystal_dir, f"{stem}.pt")
        write_pe_pt(pt_path, table)

    timing = CrystalTiming(
        t_preprocess_s=t_preprocess,
        t_predict_s=t_predict,
        t_postprocess_s=time.perf_counter() - t_post0,
    )

    meta = {
        "folder": folder,
        "config": config_key,
        "domain": domain,
        "architecture": job["architecture"],
        "mode": "export",
        "n_atoms_full": int(len(atom_ids)),
        "n_atoms_graph": int(data.num_nodes),
        "n_atoms_in_graph_flag": int(table.in_graph.sum()),
        "identity_pred": bool(identity_pred),
        "csv_path": csv_path,
        "pt_path": pt_path,
        **asdict(timing),
        "t_total_s": timing.t_total_s,
        "mae_abs_eV": float(np.mean(np.abs(table.pe_pred - table.pe_true))),
        "mae_in_graph_abs_eV": float(
            np.mean(
                np.abs(
                    table.pe_pred[table.in_graph == 1]
                    - table.pe_true[table.in_graph == 1]
                )
            )
        )
        if table.in_graph.any()
        else None,
    }
    with open(
        os.path.join(crystal_dir, f"{stem}_timing.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _progress_mae_key(mode: str) -> str:
    return "mae_residual_eV" if mode == "inference" else "mae_abs_eV"


def run_job(
    job_name: str,
    mode: str,
    output_root: str,
    inference_root: Optional[str],
    checkpoint: Optional[str],
    identity_pred: bool,
    limit: Optional[int],
    device_str: str,
    write_pt: bool,
    skip_missing_checkpoint: bool,
    jobs: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    job_map = jobs or JOBS
    if job_name not in job_map:
        raise SystemExit(f"Unknown job {job_name!r}; choose from {list(job_map)}")
    if mode not in {"export", "inference", "restore"}:
        raise SystemExit(f"Unknown mode {mode!r}")
    if mode == "restore" and not inference_root:
        raise SystemExit("--mode restore requires --inference-root")

    job = dict(job_map[job_name])
    if checkpoint:
        job["checkpoint"] = checkpoint

    dataset_path = job["dataset"]
    if not os.path.isfile(dataset_path):
        raise SystemExit(f"Dataset not found: {dataset_path}")

    device = torch.device(
        device_str
        if device_str != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[{job_name}] mode={mode} device={device}")
    print(f"[{job_name}] dataset={dataset_path}")

    dataset: List[Data] = torch.load(dataset_path, weights_only=False)
    if limit is not None:
        dataset = dataset[: max(0, int(limit))]
    print(f"[{job_name}] crystals={len(dataset)}")

    model = None
    target_mean = None
    target_std = None
    ckpt_path = job["checkpoint"]
    need_model = mode in {"export", "inference"}
    if need_model:
        if identity_pred:
            print(f"[{job_name}] identity-pred: using graph.y (residual GT) as dPE")
        else:
            if not os.path.isfile(ckpt_path):
                msg = f"Checkpoint not found: {ckpt_path}"
                if skip_missing_checkpoint:
                    print(f"[{job_name}] SKIP - {msg}")
                    return
                raise SystemExit(
                    msg
                    + " (retrain/save residual model, or pass --identity-pred to "
                    "test the restore pipeline)"
                )
            print(f"[{job_name}] checkpoint={ckpt_path}")
            model, target_mean, target_std = load_residual_model(
                ckpt_path, dataset, job["architecture"], device
            )
    else:
        print(f"[{job_name}] restore from inference_root={inference_root}")

    out_root = os.path.join(output_root, job["output_subdir"])
    os.makedirs(out_root, exist_ok=True)

    summaries: List[Dict] = []
    mae_key = _progress_mae_key(mode)
    for i, data in enumerate(dataset):
        try:
            if mode == "inference":
                meta = infer_one_crystal(
                    data=data,
                    job=job,
                    out_root=out_root,
                    device=device,
                    model=model,
                    target_mean=target_mean,
                    target_std=target_std,
                    identity_pred=identity_pred,
                )
            elif mode == "restore":
                meta = restore_one_crystal(
                    data=data,
                    job=job,
                    out_root=out_root,
                    inference_root=inference_root,  # type: ignore[arg-type]
                    write_pt=write_pt,
                )
            else:
                meta = export_one_crystal(
                    data=data,
                    job=job,
                    out_root=out_root,
                    device=device,
                    model=model,
                    target_mean=target_mean,
                    target_std=target_std,
                    identity_pred=identity_pred,
                    write_pt=write_pt,
                )
        except Exception as err:
            print(f"  [{i + 1}/{len(dataset)}] FAIL {getattr(data, 'folder', '?')}: {err}")
            summaries.append(
                {
                    "folder": str(getattr(data, "folder", "")),
                    "error": str(err),
                }
            )
            continue
        summaries.append(meta)
        if (i + 1) % 25 == 0 or i == 0 or i + 1 == len(dataset):
            mae_val = meta.get(mae_key)
            mae_str = f"{mae_val:.4e}" if mae_val is not None else "n/a"
            print(
                f"  [{i + 1}/{len(dataset)}] {meta['folder']}/{meta['config']} "
                f"MAE={mae_str} eV  "
                f"t={meta['t_total_s'] * 1000:.1f} ms "
                f"(pre {meta['t_preprocess_s'] * 1000:.1f} / "
                f"pred {meta['t_predict_s'] * 1000:.1f} / "
                f"post {meta['t_postprocess_s'] * 1000:.1f})"
            )

    ok = [s for s in summaries if "error" not in s]
    summary_name = {
        "export": "export_summary.json",
        "inference": "inference_summary.json",
        "restore": "restore_summary.json",
    }[mode]
    summary_path = os.path.join(out_root, summary_name)

    mean_mae = None
    if ok and mae_key in ok[0]:
        mean_mae = float(np.mean([s[mae_key] for s in ok if mae_key in s]))

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "job": job_name,
                "mode": mode,
                "domain": job["domain"],
                "architecture": job["architecture"],
                "dataset": dataset_path,
                "checkpoint": None
                if (identity_pred or mode == "restore")
                else ckpt_path,
                "identity_pred": identity_pred if need_model else False,
                "inference_root": inference_root if mode == "restore" else None,
                "device": str(device),
                "n_requested": len(dataset),
                "n_ok": len(ok),
                "n_failed": len(summaries) - len(ok),
                f"mean_{mae_key}": mean_mae,
                "mean_t_preprocess_s": float(np.mean([s["t_preprocess_s"] for s in ok]))
                if ok
                else None,
                "mean_t_predict_s": float(np.mean([s["t_predict_s"] for s in ok]))
                if ok
                else None,
                "mean_t_postprocess_s": float(
                    np.mean([s["t_postprocess_s"] for s in ok])
                )
                if ok
                else None,
                "crystals": summaries,
            },
            fh,
            indent=2,
        )
    print(f"[{job_name}] wrote {len(ok)} crystals -> {out_root}")
    print(f"[{job_name}] summary -> {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crystal PE export / cluster inference / local restore "
            "(atom_id, pe_initial, pe_true, pe_pred) with timings."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="export",
        choices=["export", "inference", "restore", "restore-merged"],
        help=(
            "export=dumps+model→CSV; inference=model only; "
            "restore=one model; restore-merged=CGCNN+Transformer in one CSV"
        ),
    )
    parser.add_argument(
        "--job",
        action="append",
        choices=list(JOBS) + ["all"],
        help="Job name; repeatable. Use 'all' for the four combos.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=os.path.join(ROOT, "predictions"),
        help="Root directory for mirrored prediction trees.",
    )
    parser.add_argument(
        "--inference-root",
        type=str,
        default=None,
        help=(
            "Root of a prior --mode inference tree (contains point_cgcnn/, …). "
            "Required for --mode restore."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path (only valid with a single --job).",
    )
    parser.add_argument(
        "--identity-pred",
        action="store_true",
        help=(
            "Use residual ground-truth y as ΔPE to validate restore/write "
            "without a model (MAE should be ~0 outside fill policy)."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Only first N crystals.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda",
    )
    parser.add_argument(
        "--write-pt",
        action="store_true",
        help="Also write dataset-style .pt tensors beside each CSV (export/restore).",
    )
    parser.add_argument(
        "--skip-missing-checkpoint",
        action="store_true",
        help="Skip jobs whose default checkpoint file is absent.",
    )
    parser.add_argument(
        "--job-profile",
        type=str,
        default="default",
        choices=["default", "global_v2"],
        help="Checkpoint set: default=original delivery, global_v2=predictions_new.",
    )
    parser.add_argument(
        "--split-json",
        type=str,
        default=None,
        help=(
            "Path to delivery_split_indices.json for restore-merged split flags "
            "(avoids recomputing val indices)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = get_jobs(args.job_profile)
    jobs_arg = args.job or ["all"]
    if "all" in jobs_arg:
        selected = list(jobs)
    else:
        selected = list(dict.fromkeys(jobs_arg))

    if args.checkpoint and len(selected) != 1:
        raise SystemExit("--checkpoint requires exactly one --job")
    if args.mode in {"restore", "restore-merged"} and not args.inference_root:
        raise SystemExit(f"--mode {args.mode} requires --inference-root")

    if args.mode == "restore-merged":
        domains: List[str] = []
        for name in selected:
            domain = jobs[name]["domain"]
            if domain not in domains:
                domains.append(domain)
        split_stats: Dict[str, Dict[str, int]] = {}
        for domain in domains:
            split_stats[domain] = run_merged_restore(
                domain=domain,
                output_root=args.output_root,
                inference_root=args.inference_root,
                limit=args.limit,
                split_json=args.split_json,
                jobs=jobs,
            )
        point = split_stats.get("point", {"n_train": 0, "n_val": 0})
        planar = split_stats.get("planar", {"n_train": 0, "n_val": 0})
        info_path = write_split_info(
            args.output_root,
            seed=DELIVERY_SPLIT_SEED,
            val_fraction=DELIVERY_VAL_FRACTION,
            point_n_train=int(point["n_train"]),
            point_n_val=int(point["n_val"]),
            planar_n_train=int(planar["n_train"]),
            planar_n_val=int(planar["n_val"]),
            split_json=args.split_json,
        )
        readme_path = write_variables_readme(args.output_root)
        print(f"[restore-merged] variables -> {readme_path}")
        print(f"[restore-merged] split info -> {info_path}")
        return

    for name in selected:
        run_job(
            job_name=name,
            mode=args.mode,
            output_root=args.output_root,
            inference_root=args.inference_root,
            checkpoint=args.checkpoint,
            identity_pred=args.identity_pred,
            limit=args.limit,
            device_str=args.device,
            write_pt=args.write_pt,
            skip_missing_checkpoint=args.skip_missing_checkpoint,
            jobs=jobs,
        )


if __name__ == "__main__":
    main()
