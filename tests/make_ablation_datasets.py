import argparse
import os
from typing import List

import torch
from torch_geometric.data import Data

NODE_FEATURES = [
    "type",
    "per_atom_pe",
    "is_defect",
    "dist_to_defect",
]

EDGE_FEATURES = [
    "distance",
    "same_type",
    "incident_defect",
]


def _drop_feature_from_data(
    data: Data,
    node_feature_idx: int | None = None,
    edge_feature_idx: int | None = None,
) -> Data:
    new_data = data.clone()

    if node_feature_idx is not None:
        if new_data.x is None:
            raise ValueError("Data.x is missing; cannot drop node feature.")
        if new_data.x.dim() != 2:
            raise ValueError(f"Expected node features with 2 dims, got {new_data.x.dim()}.")
        if node_feature_idx >= new_data.x.size(1):
            raise ValueError(
                f"Node feature index {node_feature_idx} out of range "
                f"for {new_data.x.size(1)} features."
            )
        keep = [i for i in range(new_data.x.size(1)) if i != node_feature_idx]
        new_data.x = new_data.x[:, keep]

    if edge_feature_idx is not None:
        if new_data.edge_attr is None:
            raise ValueError("Data.edge_attr is missing; cannot drop edge feature.")
        if new_data.edge_attr.dim() != 2:
            raise ValueError(
                f"Expected edge features with 2 dims, got {new_data.edge_attr.dim()}."
            )
        if edge_feature_idx >= new_data.edge_attr.size(1):
            raise ValueError(
                f"Edge feature index {edge_feature_idx} out of range "
                f"for {new_data.edge_attr.size(1)} features."
            )
        keep = [i for i in range(new_data.edge_attr.size(1)) if i != edge_feature_idx]
        new_data.edge_attr = new_data.edge_attr[:, keep]

    return new_data


def _build_variant_dataset(
    dataset: List[Data],
    node_feature_idx: int | None = None,
    edge_feature_idx: int | None = None,
) -> List[Data]:
    return [
        _drop_feature_from_data(
            data,
            node_feature_idx=node_feature_idx,
            edge_feature_idx=edge_feature_idx,
        )
        for data in dataset
    ]


def generate_ablation_datasets(input_path: str, output_dir: str) -> None:
    # The dataset stores PyG Data objects; allow full unpickling for trusted inputs.
    dataset = torch.load(input_path, weights_only=False)
    if not isinstance(dataset, list):
        raise ValueError(f"Expected a list of Data objects, got {type(dataset)}.")

    os.makedirs(output_dir, exist_ok=True)

    for idx, name in enumerate(NODE_FEATURES):
        variant = _build_variant_dataset(dataset, node_feature_idx=idx)
        output_path = os.path.join(output_dir, f"pyg_dataset_no_node_{name}.pt")
        torch.save(variant, output_path)
        print(f"Saved {len(variant)} graphs to {output_path}")

    for idx, name in enumerate(EDGE_FEATURES):
        variant = _build_variant_dataset(dataset, edge_feature_idx=idx)
        output_path = os.path.join(output_dir, f"pyg_dataset_no_edge_{name}.pt")
        torch.save(variant, output_path)
        print(f"Saved {len(variant)} graphs to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ablation datasets by dropping one feature at a time."
    )
    parser.add_argument(
        "--input",
        default="pyg_dataset.pt",
        help="Path to the full PyG dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets",
        help="Directory to write ablation datasets.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)
    generate_ablation_datasets(input_path, output_dir)
