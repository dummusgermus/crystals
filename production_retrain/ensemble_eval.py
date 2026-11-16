"""Delivery val set: old production vs. single retrained seeds vs. mean ensemble of the 3 retrained seeds.
Writes production_retrain/ensemble_eval.{md,json}."""
import json, os, sys
import numpy as np
import torch
from torch_geometric.loader import DataLoader
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from atom_encodings import encode_dataset
from delivery_global_v2 import CHECKPOINTS, TOTALS_DATASETS, _build_model
from edge_features import encode_edges
from train_single import ensure_graph_delta_field, select_total_targets
sys.path.insert(0, os.path.join(ROOT, "ensemble_study"))
from analyze import evaluate

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preds(path, domain, kind, val_idx):
    ck = torch.load(path, map_location=dev, weights_only=False)
    ds = torch.load(TOTALS_DATASETS[domain], weights_only=False); ensure_graph_delta_field(ds)
    ds = encode_edges(encode_dataset(ds, ck.get("atom_encoding", "type12")), ck["config"].get("edge", "raw"))
    m = _build_model(kind, ds, dict(ck["config"])).to(dev); m.load_state_dict(ck["model_state"]); m.eval()
    P, Y, G, TG, TF, off = [], [], [], [], [], 0
    with torch.no_grad():
        for b in DataLoader([ds[i] for i in val_idx], batch_size=32, shuffle=False):
            b = b.to(dev)
            P.append((m(b) * float(ck["target_std"]) + float(ck["target_mean"])).view(-1).double().cpu())
            Y.append(b.y.view(-1).double().cpu()); G.append((b.batch + off).cpu()); off += b.num_graphs
            TG.append(select_total_targets(b, target_mode="graph").double().cpu()); TF.append(select_total_targets(b, target_mode="full").double().cpu())
    return torch.cat(P).numpy(), {"y": torch.cat(Y), "graph": torch.cat(G), "tot_graph": torch.cat(TG), "tot_full": torch.cat(TF)}

out, L = {}, ["# Delivery val set: old vs. retrained single seeds vs. 3- and 5-seed mean ensembles\n",
              "| model | method | MAE (meV) | R_tot graph % | R_tot full % | |dE| graph (eV) |", "|---|---|---|---|---|---|"]
for key in ["point_cgcnn", "planar_cgcnn", "point_transformer", "planar_transformer"]:
    domain, kind = key.split("_", 1)
    old = torch.load(CHECKPOINTS[key], map_location="cpu", weights_only=False); vi = old["val_indices"]
    po, d = preds(CHECKPOINTS[key], domain, kind, vi)
    pn = [preds(os.path.join(ROOT, "production_retrain", "ckpts", f"{key}_s{s}.pt"), domain, kind, vi)[0] for s in range(5)]
    rows = {"old": evaluate(po, d), **{f"new s{s}": evaluate(p, d) for s, p in enumerate(pn)}, "new ensemble (3)": evaluate(np.mean(pn[:3], 0), d), "new ensemble (5)": evaluate(np.mean(pn, 0), d)}
    out[key] = rows
    for meth, v in rows.items():
        L.append(f"| {key} | {meth} | {v['mae']*1000:.2f} | {v['rtot_graph']:.2f} | {v['rtot_full']:.2f} | {v['dE_graph']:.3f} |")
json.dump(out, open(os.path.join(ROOT, "production_retrain", "ensemble_eval.json"), "w"), indent=1)
open(os.path.join(ROOT, "production_retrain", "ensemble_eval.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
