"""Alternative atom-type encodings for the point / planar Laves datasets.

Current datasets store ``x[:, 0] = type`` with LAMMPS type 1/2.  Type 1 is the
A species (1/3 of atoms, 8a site in C15), type 2 the B species, so the label
encodes the *site role* consistently, but the actual chemistry (``data.z``,
atomic numbers) is invisible to the model: Ag-Be and Ba-Al look identical and
the scalar 1/2 imposes a fake ordinal relation.

Each encoding replaces column 0 of ``x`` with a new block; all other node
features (initial PE, defect flag, distance, cycle counts) are kept unchanged.

Encodings
---------
type12          : unchanged baseline (scalar 1.0 / 2.0)
none            : drop the type column entirely (ablation)
role_onehot     : [is_A, is_B] one-hot of the site role (type 1 / 2)
elem_onehot     : one-hot over the 61 elements present (= learned embedding)
elem_props      : 11 z-scored element descriptors (Z, period, group,
                  Mendeleev no., mass, Pauling EN, atomic radius, molar
                  volume, melting point, 1st IE, electron affinity)
elem_props_role : elem_props + role one-hot
pair_props      : own descriptors + descriptors of the *other* species in the
                  cell + role one-hot (chemical context of the AB2 compound)
onehot_props    : elem_onehot + elem_props + role one-hot (memorisation of known
                  elements + descriptors for transfer)
pair_props_edge : pair_props + two element-aware edge features:
                  d_ij / (r_i + r_j)  (bond length vs. atomic-radius sum) and
                  |EN_i - EN_j|
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
ELEMENT_JSON = os.path.join(ROOT, "element_properties.json")

DESCRIPTOR_KEYS = (
    "Z",
    "row",
    "group",
    "mendeleev_no",
    "atomic_mass",
    "electronegativity",
    "atomic_radius",
    "molar_volume",
    "melting_point",
    "ionization_energy_1",
    "electron_affinity",
)

ENCODINGS = (
    "type12",
    "none",
    "role_onehot",
    "elem_onehot",
    "elem_props",
    "elem_props_role",
    "pair_props",
    "pair_props_edge",
    "onehot_props",
)


class ElementTable:
    def __init__(self, path: str = ELEMENT_JSON) -> None:
        with open(path, encoding="utf-8") as fh:
            elements = json.load(fh)["elements"]
        self.by_z: Dict[int, dict] = {int(v["Z"]): v for v in elements.values()}
        self.z_vocab: List[int] = sorted(self.by_z)
        self.z_to_idx = {z: i for i, z in enumerate(self.z_vocab)}
        max_z = max(self.z_vocab) + 1
        raw = np.array(
            [[float(self.by_z[z][k]) for k in DESCRIPTOR_KEYS] for z in self.z_vocab]
        )
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std < 1e-8] = 1.0
        normed = (raw - mean) / std
        # Dense lookup tables indexed by atomic number.
        self.desc = torch.zeros((max_z, len(DESCRIPTOR_KEYS)), dtype=torch.float)
        self.onehot_idx = torch.full((max_z,), -1, dtype=torch.long)
        self.radius = torch.zeros(max_z, dtype=torch.float)
        self.en = torch.zeros(max_z, dtype=torch.float)
        for i, z in enumerate(self.z_vocab):
            self.desc[z] = torch.tensor(normed[i], dtype=torch.float)
            self.onehot_idx[z] = i
            self.radius[z] = float(self.by_z[z]["atomic_radius"])
            self.en[z] = float(self.by_z[z]["electronegativity"])

    def onehot(self, z: torch.Tensor) -> torch.Tensor:
        idx = self.onehot_idx[z]
        if (idx < 0).any():
            missing = sorted(set(z[idx < 0].tolist()))
            raise KeyError(f"Atomic numbers missing from element table: {missing}")
        return torch.nn.functional.one_hot(idx, len(self.z_vocab)).float()


_TABLE: ElementTable | None = None


def element_table() -> ElementTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = ElementTable()
    return _TABLE


def _role_onehot(types: torch.Tensor) -> torch.Tensor:
    t = types.round().long()
    if not bool(((t == 1) | (t == 2)).all()):
        raise ValueError(f"Unexpected type labels: {sorted(set(t.tolist()))}")
    return torch.stack([(t == 1).float(), (t == 2).float()], dim=-1)


def _partner_z(z: torch.Tensor) -> torch.Tensor:
    """Atomic number of the other species in the (binary) cell."""
    species = sorted(set(z.tolist()))
    if len(species) == 1:
        return z.clone()
    if len(species) != 2:
        raise ValueError(f"Expected binary cell, got species {species}")
    a, b = species
    return torch.where(z == a, torch.full_like(z, b), torch.full_like(z, a))


def encode_graph(data, encoding: str):
    """Return a cloned graph with column 0 of ``x`` re-encoded."""
    if encoding not in ENCODINGS:
        raise ValueError(f"Unknown encoding {encoding!r}; choose from {ENCODINGS}")
    out = data.clone()
    if encoding == "type12":
        return out
    tab = element_table()
    types = data.x[:, 0]
    rest = data.x[:, 1:]
    z = data.z.long().view(-1)

    if encoding == "none":
        block = rest.new_zeros((rest.size(0), 0))
    elif encoding == "role_onehot":
        block = _role_onehot(types)
    elif encoding == "elem_onehot":
        block = tab.onehot(z)
    elif encoding == "elem_props":
        block = tab.desc[z]
    elif encoding == "elem_props_role":
        block = torch.cat([tab.desc[z], _role_onehot(types)], dim=-1)
    elif encoding == "onehot_props":
        block = torch.cat([tab.onehot(z), tab.desc[z], _role_onehot(types)], dim=-1)
    elif encoding in ("pair_props", "pair_props_edge"):
        pz = _partner_z(z)
        block = torch.cat([tab.desc[z], tab.desc[pz], _role_onehot(types)], dim=-1)
    else:  # pragma: no cover
        raise AssertionError(encoding)

    out.x = torch.cat([block, rest], dim=-1)

    if encoding == "pair_props_edge":
        src, dst = data.edge_index
        dist = data.edge_attr[:, 0]
        rsum = tab.radius[z[src]] + tab.radius[z[dst]]
        strain = dist / rsum.clamp(min=1e-6)
        den = (tab.en[z[src]] - tab.en[z[dst]]).abs()
        out.edge_attr = torch.cat(
            [data.edge_attr, strain.unsqueeze(-1), den.unsqueeze(-1)], dim=-1
        )
    return out


def encode_dataset(dataset, encoding: str) -> list:
    return [encode_graph(d, encoding) for d in dataset]


def element_pair_key(data) -> str:
    """Chemistry key (A-B element pair) used for chemistry-held-out splits."""
    if hasattr(data, "element_a") and hasattr(data, "element_b"):
        return f"{data.element_a}-{data.element_b}"
    folder = str(getattr(data, "folder", ""))
    if "_" in folder:
        return folder.split("_", 1)[1]
    species = sorted(set(data.z.view(-1).tolist()))
    return "-".join(str(s) for s in species)
