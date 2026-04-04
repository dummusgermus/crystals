"""Add a master node to the cycle34 dataset.

Implements the "master node" idea from Gilmer et al. (2017) – Neural Message
Passing for Quantum Chemistry (Section 5.2, Virtual Graph Elements).

For each graph in cycle34_dataset.pt the script:
  - Appends an ``is_master`` indicator (0/1) to every node feature vector.
  - Adds one virtual master node (connected to all real nodes) whose features
    are zero everywhere except ``is_master = 1``.
  - Appends an ``is_master_edge`` indicator (0/1) to every edge feature vector.
    Edges between the master node and real nodes have ``is_master_edge = 1``;
    original edges keep ``is_master_edge = 0``.
  - Stores edges to the master node in a single orientation (node → master);
    the bidirectional model doubles them in the forward pass.
  - Sets the master node target ``y = 0`` and adds a boolean ``node_mask``
    attribute (True for real nodes) so the master node can be excluded from
    loss computation during training.

Saves the result as ``adv_datasets/masternode_cycle34_dataset.pt``.
"""

from __future__ import annotations

import os
from typing import List

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


def add_master_node(data: Data) -> Data:
    """Return a copy of *data* augmented with a master node."""
    n = data.num_nodes
    master_idx = n
    feat_dim = data.x.size(-1)
    edge_feat_dim = data.edge_attr.size(-1)
    num_edges = data.edge_index.size(1) if data.edge_index.numel() > 0 else 0

    # -- node features: [original | is_master] --------------------------------
    real_flag = torch.zeros(n, 1, dtype=torch.float)
    master_row = torch.cat(
        [torch.zeros(1, feat_dim, dtype=torch.float), torch.ones(1, 1, dtype=torch.float)],
        dim=-1,
    )
    new_x = torch.cat([torch.cat([data.x, real_flag], dim=-1), master_row], dim=0)

    # -- edge features: [original | is_master_edge] ---------------------------
    if num_edges > 0:
        orig_flag = torch.zeros(num_edges, 1, dtype=torch.float)
        new_edge_attr = torch.cat([data.edge_attr, orig_flag], dim=-1)
    else:
        new_edge_attr = torch.zeros((0, edge_feat_dim + 1), dtype=torch.float)

    # One edge per real node → master (bidirectional wrapper adds the reverse)
    node_ids = torch.arange(n, dtype=torch.long)
    master_ids = torch.full((n,), master_idx, dtype=torch.long)
    master_ei = torch.stack([node_ids, master_ids], dim=0)

    master_ea = torch.cat(
        [torch.zeros(n, edge_feat_dim, dtype=torch.float), torch.ones(n, 1, dtype=torch.float)],
        dim=-1,
    )

    new_edge_index = torch.cat([data.edge_index, master_ei], dim=1) if num_edges > 0 else master_ei
    new_edge_attr = torch.cat([new_edge_attr, master_ea], dim=0)

    # -- targets: master node gets y=0 (masked during training) ---------------
    new_y = torch.cat([data.y, torch.zeros(1, data.y.size(-1), dtype=torch.float)], dim=0)

    # -- node_mask: True for real nodes, False for master ---------------------
    node_mask = torch.ones(n + 1, dtype=torch.bool)
    node_mask[master_idx] = False

    # -- position: place master node at centroid ------------------------------
    new_pos = None
    if data.pos is not None:
        centroid = data.pos.mean(dim=0, keepdim=True)
        new_pos = torch.cat([data.pos, centroid], dim=0)

    new_data = Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        pos=new_pos,
        y=new_y,
    )
    new_data.node_mask = node_mask

    for attr in _METADATA_ATTRS:
        if hasattr(data, attr):
            setattr(new_data, attr, getattr(data, attr))

    return new_data


def build_masternode_dataset(input_path: str, output_path: str) -> None:
    """Load *input_path*, add master nodes, and save to *output_path*."""
    print(f"Loading dataset from {input_path} …")
    dataset: List[Data] = torch.load(input_path, weights_only=False)
    print(f"  {len(dataset)} graphs loaded")

    print("Adding master nodes …")
    new_dataset: List[Data] = []
    for i, data in enumerate(dataset):
        new_dataset.append(add_master_node(data))
        if (i + 1) % 100 == 0 or i + 1 == len(dataset):
            print(f"  [{i + 1}/{len(dataset)}] graphs processed")

    save_dataset(new_dataset, output_path)

    sample = new_dataset[0]
    print(f"Saved {len(new_dataset)} graphs to {output_path}")
    print(f"  Node feature dim : {sample.x.size(-1)} (was {sample.x.size(-1) - 1})")
    print(f"  Edge feature dim : {sample.edge_attr.size(-1)} (was {sample.edge_attr.size(-1) - 1})")
    print(f"  Nodes per graph  : {sample.num_nodes} (was {sample.num_nodes - 1})")


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(root_dir, "adv_datasets", "cycle34_dataset.pt")
    output_file = os.path.join(root_dir, "adv_datasets", "masternode_cycle34_dataset.pt")
    build_masternode_dataset(input_file, output_file)
    print("Done.")
