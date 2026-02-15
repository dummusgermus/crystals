import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ovito.io import import_file

FILENAME_RE = re.compile(
    r"^(relaxed|unrelaxed)_(\d+)-(\d+)-(\d+)-([^.]+)\.(data|dump)$"
)

PE_PROPERTY_CANDIDATES = [
    "c_pe_potential_energy",
    "c_pe_potential_energy[1]",
    "pe_potential_energy",
]


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


def _iter_valid_simulation_folders(simulations_dir: str) -> List[str]:
    folders: List[str] = []
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
        folders.append(folder_path)
    return folders


def _max_distance_for_pair(
    unrelaxed_dump_path: str,
    relaxed_dump_path: str,
    defect_id: int,
    threshold: float,
    use_absolute: bool,
) -> Tuple[Optional[float], int]:
    pipeline = import_file(unrelaxed_dump_path)
    data = pipeline.compute()
    particle_ids = _get_particle_ids(data)
    id_to_index = {int(pid): idx for idx, pid in enumerate(particle_ids)}
    if defect_id not in id_to_index:
        raise ValueError(f"Defect id {defect_id} not found in {unrelaxed_dump_path}")
    defect_index = id_to_index[defect_id]

    positions = np.asarray(data.particles.positions)
    per_atom_pe = _get_property_array(data.particles, PE_PROPERTY_CANDIDATES)
    if per_atom_pe is None:
        raise ValueError(f"Missing per-atom potential energy in {unrelaxed_dump_path}")

    relaxed_pipeline = import_file(relaxed_dump_path)
    relaxed_data = relaxed_pipeline.compute()
    relaxed_ids = _get_particle_ids(relaxed_data)
    relaxed_pe = _get_property_array(relaxed_data.particles, PE_PROPERTY_CANDIDATES)
    if relaxed_pe is None:
        raise ValueError(f"Missing per-atom potential energy in {relaxed_dump_path}")

    relaxed_pe_by_id = {
        int(pid): float(relaxed_pe[idx]) for idx, pid in enumerate(relaxed_ids)
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

    def _min_image_distance(i_idx: int, j_idx: int) -> float:
        dfrac = frac_positions[j_idx] - frac_positions[i_idx]
        for dim in range(3):
            if pbc[dim]:
                dfrac[dim] -= np.round(dfrac[dim])
        dvec = dfrac @ cell_matrix
        return float(np.linalg.norm(dvec))

    max_distance = None
    missing_relaxed = 0
    for idx, pid in enumerate(particle_ids):
        relaxed_value = relaxed_pe_by_id.get(int(pid))
        if relaxed_value is None:
            missing_relaxed += 1
            continue
        delta = relaxed_value - float(per_atom_pe[idx])
        if use_absolute:
            delta = abs(delta)
        if delta <= threshold:
            continue
        dist = _min_image_distance(defect_index, idx)
        if max_distance is None or dist > max_distance:
            max_distance = dist

    return max_distance, missing_relaxed


def collect_max_distances(
    simulations_dir: str,
    threshold: float,
    use_absolute: bool,
) -> Tuple[np.ndarray, Dict[str, int]]:
    max_distances: List[float] = []
    stats = {
        "folders_considered": 0,
        "pairs_used": 0,
        "pairs_without_hits": 0,
        "missing_relaxed_atoms": 0,
    }

    for folder_path in _iter_valid_simulation_folders(simulations_dir):
        stats["folders_considered"] += 1
        data_files: List[Tuple[str, Dict[str, str]]] = []
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

            max_dist, missing_relaxed = _max_distance_for_pair(
                dump_path,
                relaxed_dump_path,
                defect_id=parsed["defect_id"],
                threshold=threshold,
                use_absolute=use_absolute,
            )
            stats["missing_relaxed_atoms"] += missing_relaxed
            stats["pairs_used"] += 1
            if max_dist is None:
                stats["pairs_without_hits"] += 1
                continue
            max_distances.append(max_dist)

    return np.asarray(max_distances, dtype=float), stats


def plot_histogram(
    distances: np.ndarray,
    output_path: str,
    bins: int,
    threshold: float,
    use_absolute: bool,
    dpi: int,
) -> None:
    if distances.size == 0:
        raise ValueError("No max-distance values collected; nothing to plot.")

    plt.figure(figsize=(11, 6))
    plt.hist(
        distances,
        bins=bins,
        color="seagreen",
        edgecolor="black",
        linewidth=0.25,
    )
    label = "absolute" if use_absolute else "signed"
    plt.xlabel(f"Max distance to defect (>{threshold:g} eV, {label}) [Å]")
    plt.ylabel("Graph count")
    plt.title(f"Furthest atom above threshold across {distances.size} graphs")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Histogram of the furthest-from-defect atom with energy change "
            "above a threshold (per graph)."
        )
    )
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--simulations-dir",
        default=os.path.join(root_dir, "SIMULATIONS"),
        help="Path to the SIMULATIONS directory.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(root_dir, "energy_change_distance_histogram.png"),
        help="Output PNG path for the histogram.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.07,
        help="Energy-change threshold (eV).",
    )
    parser.add_argument(
        "--signed",
        action="store_true",
        help="Use signed energy changes (default is absolute).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for the saved PNG.",
    )
    args = parser.parse_args()

    use_absolute = not args.signed
    distances, stats = collect_max_distances(
        simulations_dir=args.simulations_dir,
        threshold=args.threshold,
        use_absolute=use_absolute,
    )
    plot_histogram(
        distances=distances,
        output_path=args.output,
        bins=args.bins,
        threshold=args.threshold,
        use_absolute=use_absolute,
        dpi=args.dpi,
    )

    print(
        "Histogram saved to "
        f"{args.output} using {distances.size} graphs out of "
        f"{stats['pairs_used']} pairs."
    )
    if stats["pairs_without_hits"] > 0:
        print(f"Pairs with no atoms above threshold: {stats['pairs_without_hits']}")
    if stats["missing_relaxed_atoms"] > 0:
        print(f"Skipped {stats['missing_relaxed_atoms']} atoms missing relaxed PE.")


if __name__ == "__main__":
    main()
