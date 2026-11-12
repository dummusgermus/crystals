"""Backfill finished atom-encoding sweep runs (results/*.json) into W&B project yara-crystals.

Idempotent: runs whose name already exists in the project are skipped.

    python atom_encoding_sweep/backfill_wandb.py [--workers 6] [--dry-run]

Logged per run: full config (same keys as live runs of train_atom_encoding_sweep.py,
rebuilt from the saved checkpoint + result JSON), per-epoch val curves
(val/mae, val/r_tot_median, val/best_*), summary = final val/test metrics at
both checkpoints. Tagged "backfilled"; lr / per-epoch |dE| were not recorded
for these runs and are therefore absent.
"""

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
PKG = "/work/log1/erik.mueller/python-packages"
WANDB_HOME = "/work/log1/erik.mueller/.wandb"
PROJECT = "yara-crystals"
GIT_AT_RUNTIME = "efe77c46 (+ uncommitted sweep code)"


def _setup_env():
    if PKG not in sys.path:
        sys.path.append(PKG)
    for var, sub in (("WANDB_DIR", ""), ("WANDB_CACHE_DIR", "cache"),
                     ("WANDB_CONFIG_DIR", "config"), ("WANDB_DATA_DIR", "data")):
        os.environ.setdefault(var, os.path.join(WANDB_HOME, sub))
        os.makedirs(os.environ[var], exist_ok=True)
    if "WANDB_API_KEY" not in os.environ:
        prof = "/work/log1/erik.mueller/.my_profile"
        for line in open(prof):
            m = re.match(r'^export WANDB_API_KEY="?([^"\s]+)"?', line)
            if m:
                os.environ["WANDB_API_KEY"] = m.group(1)


def slurm_ids():
    ids = {}
    for path in glob.glob(os.path.join(ROOT, "logs", "jara-atomenc-*_*.out")):
        m = re.search(r"jara-atomenc-(\d+)_(\d+)\.out$", path)
        for line in open(path, errors="ignore"):
            k = re.match(r"^=== (\S+) \|", line)
            if k:
                ids[k.group(1)] = (m.group(1), m.group(2))
    return ids


def _flatten(prefix, d):
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        out.update(_flatten(key, v) if isinstance(v, dict) else {key: v})
    return out


def build(path, ids):
    import torch
    from delivery_global_v2 import TOTALS_DATASETS

    r = json.load(open(path))
    key = r["run_key"]
    ck_path = os.path.join(HERE, "checkpoints", f"{key}.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False) if os.path.isfile(ck_path) else {}
    model_cfg = ck.get("config", {})
    total_cfg = ck.get("total_loss_config", {})
    cgcnn = r["model_kind"] == "cgcnn"
    job = ids.get(key, (None, None))
    config = {
        "experiment": "atom_encoding_sweep",
        "run_key": key,
        "domain": r["domain"],
        "model_kind": r["model_kind"],
        "architecture": "GatedGNN (CGCNN-style)" if cgcnn else "GraphTransformer",
        "encoding": r["encoding"],
        "split": r["split"],
        "split_fractions": "70/15/15",
        "seed": r["seed"],
        "epochs": r["epochs"],
        "checkpoint_metric": "val_r_tot_median",
        **{f"model/{k}": v for k, v in model_cfg.items()},
        "optimizer": "Adam" if cgcnn else "AdamW",
        "scheduler": "ReduceLROnPlateau(factor=0.5,patience=8,min_lr=1e-6,on=val_mae)" if cgcnn else "cosine_warmup",
        "eval_batch_size": 32,
        "atom_loss": "per_graph_mae (normalised targets)",
        **_flatten("total_loss", total_cfg or {"lambda_tot": r["lambda_tot"], "target_mode": r["total_target_mode"]}),
        "dataset": TOTALS_DATASETS[r["domain"]],
        "x_dim": r["x_dim"],
        "edge_dim": r["edge_dim"],
        "n_train": r["n_train"],
        "n_val": r["n_val"],
        "n_test": r["n_test"],
        "target_mean": ck.get("target_mean"),
        "target_std": ck.get("target_std"),
        "n_params": r["n_params"],
        "git_commit": GIT_AT_RUNTIME,
        "slurm_job_id": job[0],
        "slurm_array_task_id": job[1],
        "device": "cuda",
        "backfilled": True,
    }
    summary = {**_flatten("ckpt_rtot", r["ckpt_rtot"]), **_flatten("ckpt_mae", r["ckpt_mae"]), "wall_s": r["wall_s"]}
    return key, config, r["curves"], summary


def upload(path, ids):
    _setup_env()
    import wandb

    key, config, curves, summary = build(path, ids)
    run = wandb.init(
        project=PROJECT, name=key, config=config,
        group=f"atom-encoding-{config['split']}",
        job_type=f"{config['domain']}_{config['model_kind']}",
        tags=["atom-encoding-sweep", config["split"], config["domain"], config["model_kind"],
              config["encoding"], "backfilled"],
        reinit="finish_previous", settings=wandb.Settings(init_timeout=120, silent=True),
    )
    best_r, best_m = float("inf"), float("inf")
    for ep, (mae, rt) in enumerate(zip(curves["val_mae"], curves["val_r_tot_median"]), 1):
        best_r, best_m = min(best_r, rt), min(best_m, mae)
        run.log({"val/mae": mae, "val/r_tot_median": rt,
                 "val/best_r_tot_median": best_r, "val/best_mae": best_m}, step=ep)
    run.summary.update(summary)
    run.finish()
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    _setup_env()
    import wandb

    api = wandb.Api(timeout=60)
    existing = {run.name for run in api.runs(f"{api.default_entity}/{PROJECT}")}
    paths = sorted(glob.glob(os.path.join(HERE, "results", "*.json")))
    todo = [p for p in paths if os.path.basename(p)[:-5] not in existing]
    print(f"{len(paths)} result files, {len(paths) - len(todo)} already in W&B, {len(todo)} to upload", flush=True)
    ids = slurm_ids()
    if a.dry_run:
        k, cfg, cur, summ = build(todo[0], ids)
        print(k, json.dumps(cfg, indent=1, default=str)[:3000], len(cur["val_mae"]), list(summ)[:6])
        return
    done = failed = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(upload, p, ids): p for p in todo}
        for f in as_completed(futs):
            try:
                f.result()
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[fail] {os.path.basename(futs[f])}: {exc}", flush=True)
            if (done + failed) % 20 == 0:
                print(f"{done + failed}/{len(todo)} processed ({failed} failed)", flush=True)
    print(f"finished: {done} uploaded, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
