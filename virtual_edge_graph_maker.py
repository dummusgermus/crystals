"""Add virtual edges to the cycle34 dataset.

Implements the "virtual edge" idea from Gilmer et al. (2017) -- Neural Message
Passing for Quantum Chemistry (Section 5.2, Virtual Graph Elements).

For each graph in cycle34_dataset.pt the script:
  - Appends an ``is_virtual`` indicator (0/1) to every edge feature vector.
    Original edges have ``is_virtual = 0``; new virtual edges have ``is_virtual = 1``.
  - For every pair of nodes (i, j) with i < j that does NOT already have an
    edge, adds a virtual edge with features:
      [distance, same_type, incident_defect, is_virtual=1]
    where distance, same_type, and incident_defect are computed from the
    node features / positions, matching the real edge feature semantics.
  - Virtual edges are stored in one orientation (i → j with i < j);
    the bidirectional model doubles them in the forward pass.

Saves the result as ``adv_datasets/virtual_edge_cycle34_dataset.pt``.
"""

from __future__ import annotations

import os
from typing import List, Set, Tuple

import torch
from torch_geometric.data import Data

from graph_maker import save_dataset

_METADATA_ATTRS = (
    "relax_state",
    "defect_id",
    "from_type",
    "to_type",
    "wyckoff",
    "cutoff_k",
    "edge_k",
    "cutoff_radius",
    "edge_radius",
    "cutoff_mode",
    "meta",
)


def add_virtual_edges(data: Data) -> Data:
    """Return a copy of *data* with virtual edges for all unconnected pairs."""
    n = data.num_nodes
    edge_feat_dim = data.edge_attr.size(-1)
    num_existing = data.edge_index.size(1) if data.edge_index.numel() > 0 else 0

    existing_pairs: Set[Tuple[int, int]] = set()
    for k in range(num_existing):
        i = int(data.edge_index[0, k])
        j = int(data.edge_index[1, k])
        existing_pairs.add((min(i, j), max(i, j)))

    # Node features: [particle_type, pe, is_defect, dist_to_defect, ...]
    # We need particle_type (col 0) and is_defect (col 2) to compute edge features.
    types = data.x[:, 0]
    is_defect = data.x[:, 2]

    virtual_src: List[int] = []
    virtual_dst: List[int] = []
    virtual_attr: List[List[float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in existing_pairs:
                continue

            if data.pos is not None:
                dist = float(torch.norm(data.pos[i] - data.pos[j]))
            else:
                dist = 0.0

            same_type = 1.0 if types[i] == types[j] else 0.0
            incident = 1.0 if (is_defect[i] > 0.5 or is_defect[j] > 0.5) else 0.0

            virtual_src.append(i)
            virtual_dst.append(j)
            virtual_attr.append([dist, same_type, incident, 1.0])

    # Tag original edges with is_virtual = 0
    if num_existing > 0:
        orig_flag = torch.zeros(num_existing, 1, dtype=torch.float)
        new_edge_attr = torch.cat([data.edge_attr, orig_flag], dim=-1)
    else:
        new_edge_attr = torch.zeros((0, edge_feat_dim + 1), dtype=torch.float)

    if virtual_src:
        virt_ei = torch.tensor([virtual_src, virtual_dst], dtype=torch.long)
        virt_ea = torch.tensor(virtual_attr, dtype=torch.float)

        new_edge_index = torch.cat([data.edge_index, virt_ei], dim=1)
        new_edge_attr = torch.cat([new_edge_attr, virt_ea], dim=0)
    else:
        new_edge_index = data.edge_index.clone()

    new_data = Data(
        x=data.x.clone(),
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        pos=data.pos.clone() if data.pos is not None else None,
        y=data.y.clone(),
    )

    for attr in _METADATA_ATTRS:
        if hasattr(data, attr):
            setattr(new_data, attr, getattr(data, attr))

    return new_data


def build_virtual_edge_dataset(input_path: str, output_path: str) -> None:
    """Load *input_path*, add virtual edges, and save to *output_path*."""
    print(f"Loading dataset from {input_path} ...")
    dataset: List[Data] = torch.load(input_path, weights_only=False)
    print(f"  {len(dataset)} graphs loaded")

    sample = dataset[0]
    orig_edge_dim = sample.edge_attr.size(-1)
    orig_num_edges = sample.edge_index.size(1) if sample.edge_index.numel() > 0 else 0

    print("Adding virtual edges ...")
    new_dataset: List[Data] = []
    for i, data in enumerate(dataset):
        new_dataset.append(add_virtual_edges(data))
        if (i + 1) % 100 == 0 or i + 1 == len(dataset):
            print(f"  [{i + 1}/{len(dataset)}] graphs processed")

    save_dataset(new_dataset, output_path)

    s = new_dataset[0]
    new_num_edges = s.edge_index.size(1) if s.edge_index.numel() > 0 else 0
    print(f"Saved {len(new_dataset)} graphs to {output_path}")
    print(f"  Edge feature dim : {s.edge_attr.size(-1)} (was {orig_edge_dim})")
    print(f"  Edges per graph  : {new_num_edges} (was {orig_num_edges}) [sample 0]")
    print(f"  Virtual edges    : {new_num_edges - orig_num_edges} added [sample 0]")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(root_dir, "adv_datasets", "cycle34_dataset.pt")
    output_file = os.path.join(root_dir, "adv_datasets", "virtual_edge_cycle34_dataset.pt")
    build_virtual_edge_dataset(input_file, output_file)
    print("Done.")
