"""Deployed energy models (CGCNN / Graph Transformer, residual per-atom target, delivery_global_v2 defaults)
retrained on OUR compound splits, so they can be compared with the unified PaiNN on identical test structures.

    python displacement/energy_compare/train_gnn_split.py --model-key point_cgcnn --split-seed 0 --seed 0

* Model, data, features, loss and checkpoint selection: unchanged (delivery_global_v2.train_delivery_global_v2,
  the function production_retrain/retrain.py uses), 2000 epochs.
* Split: displacement/train_disp.py's union compound split (split_by_compound(keys=union_keys()), 70/15/15,
  rng 1000 + split seed): their train / val = our train / val compounds; our test compounds are never seen.
  Compounds outside our union key set are dropped.
* Output: ckpts/<key>_sp<k>_s<seed>.pt and preds/<key>_sp<k>_s<seed>.json, per test structure:
  folder, key (point: "<id>-<from>-<to>-<wyckoff>", our defect_key), predicted graph sum (eV), their graph-sum
  and full-cell targets. Scored against the same labels as the PaiNN in compare.py.
* W&B: project yara-crystals, group energy-compare.
Production checkpoints (CHECKPOINTS) are never touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__)); DISP = os.path.dirname(HERE); ROOT = os.path.dirname(DISP)
sys.path.insert(0, ROOT); sys.path.insert(0, DISP)

from atom_encodings import encode_dataset  # noqa: E402
from delivery_global_v2 import (ATOM_ENCODING, CGCNN_CONFIG, TARGET_NORM, TOTALS_DATASETS,  # noqa: E402
                                TRANSFORMER_CONFIG, _build_model, total_loss_config_for, train_delivery_global_v2)
from edge_features import encode_edges  # noqa: E402
from train_atom_encoding_sweep import WANDB_PROJECT_DEFAULT, WandbLogger  # noqa: E402
from train_single import ensure_graph_delta_field, set_seed  # noqa: E402


def compound(g, domain):
    if domain == "point":
        return g.folder.split("_", 1)[1]
    return f"{g.element_a}-{g.element_b}"


def struct_key(g, domain):
    if domain == "point":
        return f"{int(g.defect_id)}-{int(g.from_type)}-{int(g.to_type)}-{g.wyckoff}"
    return g.folder.split("_", 1)[1]


def our_parts(split_seed):
    """compound -> 0 train / 1 val / 2 test, exactly as train_disp.split_by_compound with union keys."""
    from train_disp import union_keys
    keys = sorted(union_keys())
    rng = np.random.default_rng(1000 + split_seed); rng.shuffle(keys)
    n1 = int(round(0.7 * len(keys))); n2 = n1 + int(round(0.15 * len(keys)))
    return {k: (0 if i < n1 else 1 if i < n2 else 2) for i, k in enumerate(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True,
                    choices=["point_cgcnn", "planar_cgcnn", "point_transformer", "planar_transformer"])
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    domain, kind = a.model_key.split("_", 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{a.model_key}_sp{a.split_seed}_s{a.seed}"
    ck = os.path.join(HERE, "ckpts", f"{tag}.pt"); os.makedirs(os.path.dirname(ck), exist_ok=True)
    os.makedirs(os.path.join(HERE, "preds"), exist_ok=True)

    raw = torch.load(TOTALS_DATASETS[domain], weights_only=False)
    part = our_parts(a.split_seed)
    idx = [[], [], []]
    for i, g in enumerate(raw):
        p = part.get(compound(g, domain))
        if p is not None:
            idx[p].append(i)
    n_drop = len(raw) - sum(map(len, idx))
    print(f"{tag}: train {len(idx[0])} val {len(idx[1])} test {len(idx[2])} (dropped {n_drop})", flush=True)

    config = dict(CGCNN_CONFIG if kind == "cgcnn" else TRANSFORMER_CONFIG)
    total_cfg = total_loss_config_for(kind, domain)
    wb = WandbLogger(
        enabled=not a.no_wandb, project=WANDB_PROJECT_DEFAULT, name=f"ecmp-{tag}",
        config={"model_key": a.model_key, "domain": domain, "model": kind, "seed": a.seed, "split_seed": a.split_seed,
                "epochs": a.epochs, "atom_encoding": ATOM_ENCODING, "target_norm": TARGET_NORM[(kind, domain)],
                "split": "displacement union compound split 70/15/15 (train_disp.split_by_compound)",
                "dataset": TOTALS_DATASETS[domain], "n_train": len(idx[0]), "n_val": len(idx[1]),
                "n_test": len(idx[2]), "n_dropped": n_drop,
                **{f"model/{k}": v for k, v in config.items()},
                **{f"total_loss/{k}": v for k, v in total_cfg.__dict__.items()},
                "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
                "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID")},
        group="energy-compare", job_type="train", tags=[domain, kind, "energy-compare"],
    )
    set_seed(a.seed)
    res = train_delivery_global_v2(domain=domain, model_kind=kind, device=device, epochs=a.epochs, metric="mae",
                                   checkpoint_path=ck,
                                   split_payload={domain: {"train_indices": idx[0], "val_indices": idx[1]}})
    for ep in range(len(res["val_mae"])):
        wb.log({"epoch": ep + 1, "train/mae": res["train_mae"][ep], "val/mae": res["val_mae"][ep],
                "val/r_tot_median": res["val_r_tot_median"][ep]}, step=ep + 1)

    # per-structure test predictions
    c = torch.load(ck, map_location=device, weights_only=False)
    ds = torch.load(TOTALS_DATASETS[domain], weights_only=False)
    ensure_graph_delta_field(ds)
    ds = encode_dataset(ds, c.get("atom_encoding", "type12"))
    ds = encode_edges(ds, c["config"].get("edge", "raw"))
    model = _build_model(kind, ds, dict(c["config"])).to(device)
    model.load_state_dict(c["model_state"]); model.eval()
    tm, ts = float(c["target_mean"]), float(c["target_std"])
    out = []
    with torch.no_grad():
        for i in idx[2]:
            g = raw[i]
            b = next(iter(DataLoader([ds[i]], batch_size=1))).to(device)
            pe = (model(b).view(-1) * ts + tm).cpu().numpy()
            out.append(dict(folder=g.folder, key=struct_key(g, domain), compound=compound(g, domain),
                            pred=float(pe.sum()), graph_true=float(g.meta["graph_delta_total_eV"]),
                            full_true=float(g.meta["delta_total_eV"]), n=int(g.num_nodes)))
    t = np.array([o["full_true"] for o in out]); p = np.array([o["pred"] for o in out]); big = np.abs(t) > 1e-3
    summ = {"test/r_tot_full_median": float(np.median(np.abs(p - t)[big] / np.abs(t[big]))), "test/n": len(out)}
    json.dump(dict(tag=tag, model_key=a.model_key, split_seed=a.split_seed, seed=a.seed, summary=summ, test=out),
              open(os.path.join(HERE, "preds", f"{tag}.json"), "w"), indent=1)
    wb.summary(summ); wb.finish()
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
