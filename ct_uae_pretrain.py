"""ct-UAE-style atomistic encoder pretrained on our own simulation data.

Implements the pretraining strategy from "ct-UAEs" (no use of their
pretrained weights): a small crystal-Transformer is trained over
``(atom_type_one_hot, cartesian_coords)`` tokens to regress per-atom
relaxed potential energies on our defect subgraphs.  The learned
``atom_embed`` ``Linear(vocab_size, emb_dim)`` layer is then saved as a
standalone "universal atomic embedding" checkpoint that can be plugged
into :mod:`gnn_models` via :class:`UAEAtomEncoder`.

Key choices (and where they differ from ``ct-UAE-main``):

* Data is taken from our own LAMMPS defect subgraphs built by
  :func:`graph_maker._build_subgraph`; element identity is recovered
  from the simulation folder name (``C15_A-B``) and the LAMMPS particle
  type (1→A, 2→B, -1→vacancy).
* A dedicated ``vacancy_token`` replaces the atom-onehot embedding for
  vacancy sites (``type == -1``), so the UAE never has to encode "no
  atom" as an implausible one-hot.
* Positional encoding is disabled (their code also leaves it commented
  out).  Random rigid augmentation of coordinates is optional.
* Readout is *per-atom*, not CLS-pooled: each token is projected to a
  scalar energy and matched against the per-node relaxed PE from our
  pipeline.  This aligns the pretraining objective with our downstream
  task.

Typical use::

    python ct_uae_pretrain.py                   # train + save embedding
    # -> writes uae_embeddings/uae_emb128.pt

    from ct_uae_pretrain import UAEAtomEncoder
    enc = UAEAtomEncoder(ckpt_path="uae_embeddings/uae_emb128.pt")
    feats = enc(data.z)                         # (N, emb_dim)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset

from graph_maker import (
    DEFECT_CUTOFF_K,
    DEFECT_CUTOFF_RADIUS,
    EDGE_CUTOFF_RADIUS,
    EDGE_K,
    SYMBOL_TO_Z,
    VACANCY_INDEX,
    VOCAB_SIZE,
    _build_subgraph,
    _parse_defect_filename,
    build_type_to_z_map,
    parse_folder_elements,
)


# ---------------------------------------------------------------------------
# Pretraining records: (Z, pos, y_per_atom) per defect subgraph
# ---------------------------------------------------------------------------

@dataclass
class PretrainRecord:
    z: torch.Tensor          # (N,) long, 0 for vacancy, otherwise atomic number
    pos: torch.Tensor        # (N, 3) float32, cartesian coords (unrelaxed)
    y: torch.Tensor          # (N,) float32, relaxed per-atom PE
    folder: str
    defect_key: str

    @property
    def num_atoms(self) -> int:
        return int(self.z.size(0))


def build_pretrain_records(
    simulations_dir: str,
    cutoff_k: int = DEFECT_CUTOFF_K,
    edge_k: int = EDGE_K,
    cutoff_radius: float = DEFECT_CUTOFF_RADIUS,
    edge_radius: float = EDGE_CUTOFF_RADIUS,
    cutoff_mode: str = "shell",
    required_file_count: int = 52,
    verbose: bool = True,
) -> List[PretrainRecord]:
    """Walk the simulations directory and build ct-UAE-ready records.

    Reuses :func:`graph_maker._build_subgraph` so the atom subset (and
    therefore the per-atom PE targets) is identical to what the GNN
    models see; adds a per-node ``z`` tensor derived from the folder
    name and the LAMMPS particle type.
    """
    records: List[PretrainRecord] = []
    skipped_folders = 0
    skipped_cases = 0
    t0 = time.time()

    for folder in sorted(os.listdir(simulations_dir)):
        if folder.endswith("_MIN"):
            continue
        folder_path = os.path.join(simulations_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        try:
            type_to_z = build_type_to_z_map(folder)
        except ValueError:
            skipped_folders += 1
            continue

        file_count = len(
            [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f))]
        )
        if file_count != required_file_count:
            skipped_folders += 1
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith(".data"):
                continue
            parsed = _parse_defect_filename(filename)
            if parsed is None or parsed["relax_state"] != "unrelaxed":
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

            try:
                x, pos, _ei, _ea, y_node, _sub, _meta = _build_subgraph(
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
            except Exception as err:
                if verbose:
                    print(f"  [skip] {folder}/{filename}: {err}")
                skipped_cases += 1
                continue

            lammps_types = x[:, 0].long().tolist()
            z_list: List[int] = []
            for t in lammps_types:
                if t == -1:
                    z_list.append(VACANCY_INDEX)
                elif t in type_to_z:
                    z_list.append(type_to_z[t])
                else:
                    z_list.append(VACANCY_INDEX)  # defensive; shouldn't happen
            z = torch.tensor(z_list, dtype=torch.long)

            records.append(PretrainRecord(
                z=z,
                pos=pos.to(torch.float32),
                y=y_node.squeeze(-1).to(torch.float32),
                folder=folder,
                defect_key=base_key,
            ))

    if verbose:
        print(f"Built {len(records)} pretrain records from {simulations_dir} "
              f"in {time.time() - t0:.1f}s "
              f"(skipped folders={skipped_folders}, cases={skipped_cases})")
    return records


# ---------------------------------------------------------------------------
# Coordinate augmentation (random rigid transform, same spirit as ct-UAE)
# ---------------------------------------------------------------------------

def _random_rotation_matrix(rng: random.Random) -> np.ndarray:
    ax = rng.uniform(-math.pi, math.pi)
    ay = rng.uniform(-math.pi, math.pi)
    az = rng.uniform(-math.pi, math.pi)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _augment_coords(
    pos: np.ndarray,
    max_translation: float,
    rng: random.Random,
) -> np.ndarray:
    """Random rotation + translation applied as a single rigid transform.

    Our coords come from PBC'd LAMMPS dumps; a rigid transform of the
    extracted subgraph preserves all pairwise distances and is exactly
    the invariance we want the transformer to learn.
    """
    t = np.array([rng.uniform(-max_translation, max_translation) for _ in range(3)])
    R = _random_rotation_matrix(rng)
    return pos @ R.T + t


# ---------------------------------------------------------------------------
# Torch Dataset + collate
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PretrainRecord],
        y_mean: float = 0.0,
        y_std: float = 1.0,
    ) -> None:
        self.records = list(records)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if y_std != 0 else 1.0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> PretrainRecord:
        r = self.records[idx]
        y_norm = (r.y - self.y_mean) / self.y_std
        return PretrainRecord(
            z=r.z, pos=r.pos, y=y_norm, folder=r.folder, defect_key=r.defect_key
        )


def make_collate(
    vocab_size: int = VOCAB_SIZE,
    augment: bool = False,
    max_translation: float = 5.0,
    seed: Optional[int] = None,
):
    """Return a collate_fn producing padded batches ready for the transformer."""
    rng = random.Random(seed)

    def collate(batch: List[PretrainRecord]):
        N_max = max(r.num_atoms for r in batch)
        B = len(batch)
        z_batch = torch.zeros(B, N_max, dtype=torch.long)
        pos_batch = torch.zeros(B, N_max, 3, dtype=torch.float32)
        y_batch = torch.zeros(B, N_max, dtype=torch.float32)
        pad_mask = torch.ones(B, N_max, dtype=torch.bool)   # True where pad
        is_vac = torch.zeros(B, N_max, dtype=torch.bool)

        for b, r in enumerate(batch):
            n = r.num_atoms
            z_batch[b, :n] = r.z
            pos = r.pos.numpy()
            if augment:
                pos = _augment_coords(pos, max_translation, rng).astype(np.float32)
            pos_batch[b, :n] = torch.from_numpy(pos.astype(np.float32))
            y_batch[b, :n] = r.y
            pad_mask[b, :n] = False
            is_vac[b, :n] = (r.z == VACANCY_INDEX)

        # one-hot on (Z-1); vacancy positions get zeroed and later replaced
        # by the vacancy token, pad positions are ignored via ``pad_mask``.
        z_clamped = z_batch.clamp(min=1, max=vocab_size)
        onehot = torch.nn.functional.one_hot(z_clamped - 1, vocab_size).to(torch.float32)
        valid = (~pad_mask) & (~is_vac)
        onehot = onehot * valid.unsqueeze(-1).to(onehot.dtype)

        return {
            "onehot": onehot,
            "pos": pos_batch,
            "pad_mask": pad_mask,
            "is_vacancy": is_vac,
            "y": y_batch,
        }

    return collate


# ---------------------------------------------------------------------------
# Crystal Transformer (our version of ct/model.py)
# ---------------------------------------------------------------------------

class CrystalTransformer(nn.Module):
    """Transformer over ``(atom_onehot, coords)`` with per-atom scalar readout."""

    def __init__(
        self,
        feature_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.1,
        use_vacancy_token: bool = True,
    ) -> None:
        super().__init__()
        if feature_size % 2 != 0:
            raise ValueError("feature_size must be divisible by 2")

        self.feature_size = feature_size
        self.emb_dim = feature_size // 2
        self.vocab_size = vocab_size
        self.use_vacancy_token = use_vacancy_token

        self.atom_embed = nn.Linear(vocab_size, self.emb_dim)
        self.coords_embed = nn.Linear(3, self.emb_dim)
        self.vacancy_token = (
            nn.Parameter(torch.zeros(self.emb_dim)) if use_vacancy_token else None
        )

        enc_layer = TransformerEncoderLayer(
            d_model=feature_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers)

        self.atom_head = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.SiLU(),
            nn.Linear(feature_size, 1),
        )

    def forward(
        self,
        onehot: torch.Tensor,
        pos: torch.Tensor,
        pad_mask: torch.Tensor,
        is_vacancy: torch.Tensor,
    ) -> torch.Tensor:
        a = self.atom_embed(onehot)
        if self.use_vacancy_token and self.vacancy_token is not None:
            vac = self.vacancy_token.view(1, 1, -1).expand_as(a)
            a = torch.where(is_vacancy.unsqueeze(-1), vac, a)
        c = self.coords_embed(pos)
        src = torch.cat([a, c], dim=-1)
        h = self.encoder(src, src_key_padding_mask=pad_mask)
        return self.atom_head(h).squeeze(-1)       # (B, N)


# ---------------------------------------------------------------------------
# Saved-UAE format + downstream encoder
# ---------------------------------------------------------------------------

def save_uae_checkpoint(
    model: CrystalTransformer,
    path: str,
    meta: Optional[dict] = None,
) -> None:
    payload = {
        "atom_embed.weight": model.atom_embed.weight.detach().cpu(),
        "atom_embed.bias": model.atom_embed.bias.detach().cpu(),
        "vocab_size": model.vocab_size,
        "emb_dim": model.emb_dim,
        "meta": meta or {},
    }
    if model.use_vacancy_token and model.vacancy_token is not None:
        payload["vacancy_token"] = model.vacancy_token.detach().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)


class UAEAtomEncoder(nn.Module):
    """Drop-in element-feature layer for downstream GNNs.

    Given a long tensor of atomic numbers ``z`` (``0`` = vacancy), returns
    the corresponding learned ``emb_dim``-dim vector.  Can be initialised
    from a saved UAE checkpoint or trained from scratch.
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        vocab_size: int = VOCAB_SIZE,
        emb_dim: int = 128,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.atom_embed = nn.Linear(vocab_size, emb_dim)
        self.vacancy_token = nn.Parameter(torch.zeros(emb_dim))

        if ckpt_path is not None:
            self.load_uae(ckpt_path)
        self.set_frozen(freeze)

    def load_uae(self, ckpt_path: str) -> None:
        sd = torch.load(ckpt_path, map_location="cpu")
        if sd["vocab_size"] != self.vocab_size or sd["emb_dim"] != self.emb_dim:
            raise ValueError(
                f"UAE checkpoint dims (vocab={sd['vocab_size']}, emb={sd['emb_dim']}) "
                f"do not match encoder (vocab={self.vocab_size}, emb={self.emb_dim})"
            )
        self.atom_embed.load_state_dict({
            "weight": sd["atom_embed.weight"],
            "bias": sd["atom_embed.bias"],
        })
        if "vacancy_token" in sd:
            self.vacancy_token.data.copy_(sd["vacancy_token"])

    def set_frozen(self, freeze: bool) -> None:
        for p in self.atom_embed.parameters():
            p.requires_grad = not freeze
        self.vacancy_token.requires_grad = not freeze

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dtype != torch.long:
            z = z.long()
        is_vac = (z == VACANCY_INDEX)
        z_safe = z.clamp(min=1, max=self.vocab_size)
        onehot = torch.nn.functional.one_hot(z_safe - 1, self.vocab_size).to(
            self.atom_embed.weight.dtype
        )
        emb = self.atom_embed(onehot)
        # Broadcasting: vacancy_token has shape (emb_dim,); is_vac has shape
        # (..., 1) after unsqueeze; both broadcast to emb's shape inside where.
        return torch.where(is_vac.unsqueeze(-1), self.vacancy_token, emb)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    feature_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1
    use_vacancy_token: bool = True

    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    warmup_epochs: int = 3
    augment: bool = True
    max_translation: float = 5.0
    val_split: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    history: List[dict] = field(default_factory=list)


def _split_records(records: Sequence[PretrainRecord], val_split: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val = max(1, int(round(val_split * len(records))))
    val_idx = set(idx[:n_val])
    train = [records[i] for i in range(len(records)) if i not in val_idx]
    val = [records[i] for i in sorted(val_idx)]
    return train, val


def _compute_y_stats(records: Sequence[PretrainRecord]) -> Tuple[float, float]:
    all_y = torch.cat([r.y for r in records])
    return float(all_y.mean()), float(all_y.std().clamp(min=1e-6))


def _run_epoch(
    model: CrystalTransformer,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    grad_clip: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_sq = 0.0
    total_n = 0
    for batch in loader:
        onehot = batch["onehot"].to(device, non_blocking=True)
        pos = batch["pos"].to(device, non_blocking=True)
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)
        is_vac = batch["is_vacancy"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        pred = model(onehot, pos, pad_mask, is_vac)
        valid = (~pad_mask)
        diff = (pred - y)[valid]
        loss = diff.pow(2).mean()

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_sq += float(diff.pow(2).sum().detach())
        total_n += int(valid.sum().detach())

    return math.sqrt(total_sq / max(total_n, 1))


def train_uae(
    records: Sequence[PretrainRecord],
    out_dir: str,
    config: TrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[CrystalTransformer, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    train_recs, val_recs = _split_records(records, config.val_split, config.seed)
    y_mean, y_std = _compute_y_stats(train_recs)
    print(f"Pretrain: {len(train_recs)} train / {len(val_recs)} val records")
    print(f"  per-atom PE stats (train): mean={y_mean:.4f} eV, std={y_std:.4f} eV")

    train_ds = PretrainDataset(train_recs, y_mean=y_mean, y_std=y_std)
    val_ds = PretrainDataset(val_recs, y_mean=y_mean, y_std=y_std)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=make_collate(augment=config.augment,
                                max_translation=config.max_translation,
                                seed=config.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=make_collate(augment=False, seed=config.seed + 1),
    )

    model = CrystalTransformer(
        feature_size=config.feature_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        use_vacancy_token=config.use_vacancy_token,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model parameters: {n_params/1e6:.2f}M on {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    total_steps = max(1, config.epochs - config.warmup_epochs)

    def lr_lambda(epoch: int) -> float:
        if epoch < config.warmup_epochs:
            return (epoch + 1) / max(1, config.warmup_epochs)
        prog = (epoch - config.warmup_epochs) / total_steps
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(out_dir, exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(out_dir, f"uae_emb{model.emb_dim}.pt")

    for epoch in range(config.epochs):
        t0 = time.time()
        train_rmse_n = _run_epoch(model, train_loader, optimizer, device, config.grad_clip)
        with torch.no_grad():
            val_rmse_n = _run_epoch(model, val_loader, None, device, 0.0)
        scheduler.step()
        # convert normalised RMSE back to physical units (eV)
        train_rmse = train_rmse_n * y_std
        val_rmse = val_rmse_n * y_std
        rec = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_rmse_eV": train_rmse,
            "val_rmse_eV": val_rmse,
            "time_s": time.time() - t0,
        }
        config.history.append(rec)
        print(
            f"  [{epoch+1:3d}/{config.epochs}] "
            f"lr={rec['lr']:.2e}  train={train_rmse:.4f} eV  val={val_rmse:.4f} eV  "
            f"({rec['time_s']:.1f}s)"
        )

        if val_rmse < best_val:
            best_val = val_rmse
            save_uae_checkpoint(
                model, best_path,
                meta={
                    "config": {
                        k: v for k, v in config.__dict__.items() if k != "history"
                    },
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "best_epoch": epoch,
                    "best_val_rmse_eV": val_rmse,
                    "num_train_records": len(train_recs),
                    "num_val_records": len(val_recs),
                },
            )

    history_path = os.path.join(out_dir, f"uae_emb{model.emb_dim}_history.json")
    with open(history_path, "w") as f:
        json.dump(
            {"best_val_rmse_eV": best_val,
             "y_mean": y_mean, "y_std": y_std,
             "history": config.history},
            f, indent=2,
        )
    print(f"  Saved best UAE checkpoint -> {best_path} (val RMSE={best_val:.4f} eV)")
    print(f"  Saved training history    -> {history_path}")
    return model, {
        "best_val_rmse_eV": best_val,
        "y_mean": y_mean,
        "y_std": y_std,
        "ckpt_path": best_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--simulations-dir", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "SIMULATIONS"))
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "uae_embeddings"))
    p.add_argument("--cutoff-k", type=int, default=DEFECT_CUTOFF_K)
    p.add_argument("--edge-k", type=int, default=EDGE_K)
    p.add_argument("--cutoff-mode", choices=["shell", "radius"], default="shell")
    p.add_argument("--feature-size", type=int, default=256,
                   help="Transformer d_model; atom_embed dim = feature_size // 2.")
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dim-feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-vacancy-token", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"Building pretrain records from {args.simulations_dir} ...")
    records = build_pretrain_records(
        simulations_dir=args.simulations_dir,
        cutoff_k=args.cutoff_k,
        edge_k=args.edge_k,
        cutoff_mode=args.cutoff_mode,
    )
    if not records:
        raise RuntimeError("No pretrain records built -- check simulations directory.")

    config = TrainConfig(
        feature_size=args.feature_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        use_vacancy_token=not args.no_vacancy_token,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        augment=not args.no_augment,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_uae(records, out_dir=args.out_dir, config=config)


if __name__ == "__main__":
    main()
