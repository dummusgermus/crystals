"""Smoke test for ct_uae_pretrain + gnn_models UAE wrappers.

Runs in <1s on CPU.  Verifies that:
  * ``CrystalTransformer`` accepts padded batches and backprops.
  * ``save_uae_checkpoint`` / ``UAEAtomEncoder`` roundtrips weights and
    reproduces embeddings bit-identically.
  * ``UAEGNNWrapper`` (via ``build_uae_gated_model_from_dataset``)
    produces per-node predictions of the expected shape, handles
    vacancies, and backprops end-to-end.

Run with::

    python test_ct_uae_smoke.py
"""

from __future__ import annotations

import os
import tempfile

import torch
from torch_geometric.data import Data

from ct_uae_pretrain import (
    VACANCY_INDEX,
    VOCAB_SIZE,
    CrystalTransformer,
    PretrainRecord,
    UAEAtomEncoder,
    make_collate,
    save_uae_checkpoint,
)
from gnn_models import build_uae_gated_model_from_dataset


def _test_transformer_collate_and_backprop() -> None:
    print("[1/3] CrystalTransformer + collate + backprop ...")
    recs = [
        PretrainRecord(
            z=torch.tensor([47, 4, 47, 4, 47], dtype=torch.long),
            pos=torch.randn(5, 3),
            y=torch.randn(5),
            folder="C15_Ag-Be",
            defect_key="1-1-1-8a",
        ),
        PretrainRecord(
            z=torch.tensor([VACANCY_INDEX, 47, 47], dtype=torch.long),
            pos=torch.randn(3, 3),
            y=torch.randn(3),
            folder="C15_Ag-Be",
            defect_key="1-1--1-8a",
        ),
    ]

    collate = make_collate(augment=True, seed=0)
    batch = collate(recs)
    assert batch["onehot"].shape == (2, 5, VOCAB_SIZE)
    assert batch["pos"].shape == (2, 5, 3)
    assert batch["pad_mask"].shape == (2, 5)
    assert batch["is_vacancy"].shape == (2, 5)
    assert batch["y"].shape == (2, 5)
    assert batch["pad_mask"][1, 3:].all() and not batch["pad_mask"][0].any()
    assert batch["is_vacancy"][1, 0].item() is True

    model = CrystalTransformer(
        feature_size=64, num_layers=2, num_heads=4,
        dim_feedforward=128, dropout=0.0,
    )
    pred = model(batch["onehot"], batch["pos"],
                 batch["pad_mask"], batch["is_vacancy"])
    assert pred.shape == (2, 5)
    loss = ((pred - batch["y"])[~batch["pad_mask"]]).pow(2).mean()
    loss.backward()
    grad_atoms = model.atom_embed.weight.grad
    assert grad_atoms is not None and grad_atoms.abs().sum().item() > 0, \
        "atom_embed weight did not receive gradient"
    print(f"      loss={loss.item():.4f}, atom_embed grad norm="
          f"{grad_atoms.norm().item():.4f}  OK")


def _test_checkpoint_roundtrip() -> None:
    print("[2/3] save_uae_checkpoint + UAEAtomEncoder roundtrip ...")
    torch.manual_seed(0)
    model = CrystalTransformer(
        feature_size=64, num_layers=1, num_heads=2,
        dim_feedforward=64, dropout=0.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "uae.pt")
        save_uae_checkpoint(model, ckpt, meta={"smoke": True})
        encoder = UAEAtomEncoder(
            ckpt_path=ckpt, vocab_size=VOCAB_SIZE,
            emb_dim=model.emb_dim, freeze=True,
        )

    z = torch.tensor([47, 4, VACANCY_INDEX, 26, 92], dtype=torch.long)
    emb_a = encoder(z)
    assert emb_a.shape == (5, model.emb_dim)

    # Replicate the forward manually to ensure bit-identical values.
    with torch.no_grad():
        onehot_ref = torch.nn.functional.one_hot(
            torch.tensor([47, 4, 1, 26, 92]) - 1, VOCAB_SIZE
        ).float()
        emb_ref = model.atom_embed(onehot_ref)
        vac_ref = torch.zeros(model.emb_dim)
        emb_ref[2] = vac_ref  # vacancy token init zeros
    assert torch.allclose(emb_a, emb_ref, atol=1e-6), \
        "Encoder output does not match reference atom_embed forward."
    print("      shapes OK, values identical to reference  OK")


def _fake_pyg_data(n: int, n_features: int, has_vacancy: bool) -> Data:
    x = torch.randn(n, n_features)
    x[:, 0] = torch.randint(-1, 3, (n,)).float()  # types 1/2/-1
    z = torch.tensor(
        [0 if t.item() == -1 else (47 if t.item() == 1 else 4) for t in x[:, 0]],
        dtype=torch.long,
    )
    if not has_vacancy:
        z = z.clamp(min=1)
    edge_src = torch.randint(0, n, (n * 2,))
    edge_dst = torch.randint(0, n, (n * 2,))
    mask = edge_src != edge_dst
    edge_index = torch.stack([edge_src[mask], edge_dst[mask]], dim=0).long()
    edge_attr = torch.randn(edge_index.size(1), 3)
    y = torch.randn(n, 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                pos=torch.randn(n, 3), y=y, z=z)


def _test_gnn_wrapper() -> None:
    print("[3/3] UAEGNNWrapper over gated GNN ...")
    dataset = [
        _fake_pyg_data(n=8, n_features=4, has_vacancy=True),
        _fake_pyg_data(n=6, n_features=4, has_vacancy=False),
    ]
    model = build_uae_gated_model_from_dataset(
        dataset,
        uae_ckpt_path=None,      # train-from-scratch UAE
        uae_emb_dim=32,
        hidden_dim=48,
        num_layers=2,
        drop_type_scalar=True,
    )
    data = dataset[0]
    saved_x = data.x.clone()
    pred = model(data)
    assert pred.shape == (data.num_nodes, 1), \
        f"Expected ({data.num_nodes}, 1), got {tuple(pred.shape)}"
    # The wrapper must leave data.x untouched for callers/batching.
    assert torch.allclose(data.x, saved_x), "data.x was not restored"
    loss = (pred - data.y).pow(2).mean()
    loss.backward()

    # UAE encoder weights must receive gradient.
    uae_weight = None
    for name, p in model.named_parameters():
        if "uae.atom_embed.weight" in name:
            uae_weight = p
            break
    assert uae_weight is not None, "UAE encoder weight not found in model"
    assert uae_weight.grad is not None and uae_weight.grad.abs().sum().item() > 0
    print(f"      pred shape OK, data.x restored, UAE grad norm="
          f"{uae_weight.grad.norm().item():.4f}  OK")


def main() -> None:
    torch.manual_seed(0)
    _test_transformer_collate_and_backprop()
    _test_checkpoint_roundtrip()
    _test_gnn_wrapper()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
