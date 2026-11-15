"""Retrain a production model with all current defaults and compare it to the
existing production checkpoint on the *same* delivery split.

    python production_retrain/retrain.py --model-key planar_cgcnn --seed 0

* Split: train / val indices are read from the existing production checkpoint,
  so old and new models see exactly the same data.
* New checkpoint -> production_retrain/ckpts/<key>_s<seed>.pt  (production
  checkpoints in CHECKPOINTS are never touched).
* Both checkpoints are evaluated on the delivery validation set (val per-atom
  MAE, median relative total-energy error with graph-sum and full-cell targets).
  Note the val set is also used for checkpoint selection in both cases.
* Logs config, per-epoch curves and the comparison to W&B (yara-crystals).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch_geometric.loader import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atom_encodings import encode_dataset  # noqa: E402
from delivery_global_v2 import (  # noqa: E402
    ATOM_ENCODING,
    CGCNN_CONFIG,
    CHECKPOINTS,
    TARGET_NORM,
    TOTALS_DATASETS,
    TRANSFORMER_CONFIG,
    _build_model,
    metric_value,
    total_loss_config_for,
    train_delivery_global_v2,
)
from edge_features import encode_edges  # noqa: E402
from train_atom_encoding_sweep import WANDB_PROJECT_DEFAULT, WandbLogger  # noqa: E402
from train_single import ensure_graph_delta_field, evaluate_with_total_energy, set_seed  # noqa: E402

OUT = os.path.join(ROOT, "production_retrain")


def evaluate_ckpt(path: str, domain: str, kind: str, val_idx, device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ds = torch.load(TOTALS_DATASETS[domain], weights_only=False)
    ensure_graph_delta_field(ds)
    ds = encode_dataset(ds, ckpt.get("atom_encoding", "type12"))
    ds = encode_edges(ds, ckpt["config"].get("edge", "raw"))
    model = _build_model(kind, ds, dict(ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    loader = DataLoader([ds[i] for i in val_idx], batch_size=32, shuffle=False)
    tm = torch.tensor(float(ckpt["target_mean"]), device=device)
    ts = torch.tensor(float(ckpt["target_std"]), device=device)
    out = {"n_params": sum(p.numel() for p in model.parameters())}
    for mode in ("graph", "full"):
        atom, tot = evaluate_with_total_energy(model, loader, device, tm, ts, total_target_mode=mode)
        out["val_mae"] = metric_value(atom, "mae")
        out[f"val_r_tot_{mode}_median"] = tot.median_rel_total_err_pct
        out[f"val_r_tot_{mode}_mean"] = tot.mean_rel_total_err_pct
        out[f"val_abs_total_err_{mode}_eV"] = tot.mean_abs_total_err_eV
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=sorted(CHECKPOINTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    domain, kind = a.model_key.split("_", 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_path = CHECKPOINTS[a.model_key]
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    split = {domain: {"train_indices": old["train_indices"], "val_indices": old["val_indices"]}}
    tag = f"{a.model_key}_s{a.seed}"
    new_path = os.path.join(OUT, "ckpts", f"{tag}.pt")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    config = dict(CGCNN_CONFIG if kind == "cgcnn" else TRANSFORMER_CONFIG)
    total_cfg = total_loss_config_for(kind, domain)
    wb = WandbLogger(
        enabled=not a.no_wandb, project=WANDB_PROJECT_DEFAULT, name=f"retrain-{tag}",
        config={"model_key": a.model_key, "domain": domain, "model": kind, "seed": a.seed,
                "epochs": a.epochs, "atom_encoding": ATOM_ENCODING, "target_norm": TARGET_NORM[(kind, domain)],
                "split": "delivery (indices from production checkpoint)", "old_checkpoint": old_path,
                "dataset": TOTALS_DATASETS[domain], "n_train": len(old["train_indices"]),
                "n_val": len(old["val_indices"]),
                **{f"model/{k}": v for k, v in config.items()},
                **{f"total_loss/{k}": v for k, v in total_cfg.__dict__.items()},
                "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
                "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID")},
        group="production-retrain", job_type="retrain", tags=[domain, kind, "retrain"],
    )
    set_seed(a.seed)
    res = train_delivery_global_v2(domain=domain, model_kind=kind, device=device, epochs=a.epochs,
                                   metric="mae", checkpoint_path=new_path, split_payload=split)
    for ep in range(len(res["val_mae"])):
        wb.log({"epoch": ep + 1, "train/mae": res["train_mae"][ep], "val/mae": res["val_mae"][ep],
                "val/r_tot_median": res["val_r_tot_median"][ep],
                "val/abs_total_err_eV": res["val_abs_total_err_eV"][ep]}, step=ep + 1)

    val_idx = old["val_indices"]
    cmp = {"old": evaluate_ckpt(old_path, domain, kind, val_idx, device),
           "new": evaluate_ckpt(new_path, domain, kind, val_idx, device)}
    out = {"model_key": a.model_key, "seed": a.seed, "epochs": a.epochs, "old_checkpoint": old_path,
           "new_checkpoint": new_path, "target_mode_new": total_cfg.target_mode,
           "target_mode_old": old.get("total_loss_config", {}).get("target_mode"), **cmp,
           "curves": {k: res[k] for k in ("train_mae", "val_mae", "val_r_tot_median")}}
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    with open(os.path.join(OUT, "results", f"{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    wb.summary({f"{w}/{k}": v for w in ("old", "new") for k, v in cmp[w].items()})
    wb.finish()
    print(json.dumps({w: cmp[w] for w in ("old", "new")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
