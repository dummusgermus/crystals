"""Evaluate the planar production models (fast single model + 5-seed ensemble) on the
GRACE C15 evaluation graphs (grace_eval/graphs_{relax,strain}.pt).

Outputs grace_eval/results.json, grace_eval/predictions.pt and grace_eval/report.md."""
import json, os, sys
import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from crystal_prediction_export import GLOBAL_V2_ENSEMBLE_JOBS, GLOBAL_V2_JOBS, load_residual_model
from atom_encodings import encode_graph
from edge_features import encode_edges_graph

OUT = os.path.dirname(os.path.abspath(__file__))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {f"{kind}-{mode}": (kind, jobs[f"planar_{kind}"]["checkpoint"])
          for kind in ("cgcnn", "transformer")
          for mode, jobs in (("fast", GLOBAL_V2_JOBS), ("ensemble5", GLOBAL_V2_ENSEMBLE_JOBS))}


def predict(graphs, kind, ckpt):
    model, mu, sd = load_residual_model(ckpt, graphs, kind, dev)
    enc = [encode_edges_graph(encode_graph(g, model.atom_encoding), model.edge_spec) for g in graphs]
    out = []
    with torch.no_grad():
        for b in DataLoader(enc, batch_size=32, shuffle=False):
            b = b.to(dev)
            p = (model(b) * sd + mu).view(-1).double().cpu()
            out += list(torch.split(p, torch.bincount(b.batch.cpu(), minlength=b.num_graphs).tolist()))
    return [o.numpy() for o in out]


def train_pairs():
    ds = torch.load(os.path.join(ROOT, "planar_pyg_dataset_residual_c14c15_totals.pt"), weights_only=False)
    return {f"{d.element_a}-{d.element_b}" for d in ds}


def main():
    relax = torch.load(os.path.join(OUT, "graphs_relax.pt"), weights_only=False)
    strain = torch.load(os.path.join(OUT, "graphs_strain.pt"), weights_only=False)
    seen = train_pairs()
    preds = {m: {"relax": predict(relax, k, c), "strain": predict(strain, k, c)} for m, (k, c) in MODELS.items()}
    torch.save(preds, os.path.join(OUT, "predictions.pt"))
    res = {"n_relax": len(relax), "n_strain": len(strain), "relax": {}, "strain": {}}

    # ---- A: relaxation at 0 % strain (our task) ----
    for struct in ("FFHFF", "Primitive", "XYZ"):
        idx = [i for i, g in enumerate(relax) if g.stack_sequence == struct]
        for subset in ("all", "seen", "unseen"):
            ii = [i for i in idx if subset == "all" or ((f"{relax[i].element_a}-{relax[i].element_b}" in seen) == (subset == "seen"))]
            if not ii:
                continue
            n = np.array([relax[i].num_nodes for i in ii])
            true_tot = np.array([float(relax[i].y.sum()) for i in ii])
            row = {"n": len(ii), "true_dE_median_meV_per_atom": float(np.median(np.abs(true_tot) / n * 1000)),
                   "baseline_zero": {"atom_mae_meV": float(np.mean([np.abs(relax[i].y.numpy()).mean() for i in ii]) * 1000),
                                     "tot_err_median_meV_per_atom": float(np.median(np.abs(true_tot) / n) * 1000),
                                     "tot_err_mean_meV_per_atom": float(np.mean(np.abs(true_tot) / n) * 1000)}}
            for m in MODELS:
                P = [preds[m]["relax"][i] for i in ii]
                pt = np.array([p.sum() for p in P]); err = np.abs(pt - true_tot) / n * 1000
                ok = np.abs(true_tot) / n * 1000 > 0.5  # R_tot only where the true change is > 0.5 meV/atom
                row[m] = {"atom_mae_meV": float(np.mean([np.abs(p - relax[i].y.numpy().ravel()).mean() for p, i in zip(P, ii)]) * 1000),
                          "tot_err_median_meV_per_atom": float(np.median(err)), "tot_err_mean_meV_per_atom": float(np.mean(err)),
                          "r_tot_median_pct": float(np.median(100 * np.abs(pt - true_tot)[ok] / np.abs(true_tot)[ok])) if ok.any() else None,
                          "n_rtot": int(ok.sum()),
                          "beats_zero_baseline_frac": float(np.mean(np.abs(pt - true_tot) < np.abs(true_tot)))}
            res["relax"][f"{struct}/{subset}"] = row

    # ---- B: strain test (colleague's expectation 2) ----
    for struct in ("FFHFF", "Primitive"):
        comps = sorted({g.element_a + g.element_b for g in strain if g.stack_sequence == struct})
        for m in ["input (no model)"] + list(MODELS):
            spreads, errs = [], {c: [] for c in ("n10", "n5", "p0", "p5", "p10")}
            for comp in comps:
                ii = [i for i, g in enumerate(strain) if g.stack_sequence == struct and g.element_a + g.element_b == comp]
                e_true = None; e_pred = []
                for i in ii:
                    g = strain[i]; n = g.num_nodes
                    e_init = g.meta["pe_initial_total_eV"]; e_true = g.meta["pe_true_total_eV"]
                    d = 0.0 if m == "input (no model)" else float(preds[m]["strain"][i].sum())
                    e_pred.append((e_init + d) / n * 1000)
                    errs[g.condition].append((e_init + d - e_true) / n * 1000)
                spreads.append(max(e_pred) - min(e_pred))
            res["strain"][f"{struct}/{m}"] = {
                "n_compounds": len(comps),
                "spread_median_meV_per_atom": float(np.median(spreads)),
                "error_mean_meV_per_atom": {c: float(np.mean(v)) for c, v in errs.items()},
                "abs_error_median_meV_per_atom": {c: float(np.median(np.abs(v))) for c, v in errs.items()}}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
