"""Loss sweep (item 2): loss_sweep/tasks_<name>.txt

    python loss_sweep/make_tasks.py <name> <seeds> [variants] [models] [extra args]

The global loss is  atom_loss + lambda * tot_loss * (atom_loss / tot_loss)  (balanced),
so lambda is the relative weight of the total-energy term (production: 0.005-0.02,
i.e. 0.5-2 % of the atom term). Variants scale the production lambda per
(model, domain), flip the total-energy target (graph sum <-> full cell) or use an
MSE instead of a Huber total-energy loss. Everything else = production default.
"""
import sys

sys.path.insert(0, ".")
from delivery_global_v2 import LAMBDA_BY_MODEL, TOTAL_TARGET_MODE_BY_MODEL  # noqa: E402

MULT = {"lam0": 0.0, "lamx0.3": 0.3, "lamx3": 3.0, "lamx10": 10.0, "lamx30": 30.0, "lamx100": 100.0}
name, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
variants = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else ["base", *MULT, "flip-target", "tot-mse"]
models = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else ["transformer", "cgcnn"]
extra = " ".join(sys.argv[5:])


def flags(v, model, dom):
    lam = LAMBDA_BY_MODEL[model][dom]
    if v == "base":
        return ""
    if v in MULT:
        return f"--lambda-tot {lam * MULT[v]:g}"
    if v == "flip-target":
        return "--tot-target " + ("graph" if TOTAL_TARGET_MODE_BY_MODEL[(model, dom)] == "full" else "full")
    if v == "tot-mse":
        return "--tot-loss mse"
    raise ValueError(v)


lines = []
for model in models:
    cmds = [f"python -u train_atom_encoding_sweep.py --split {sp} --domain {dom} --model {model} "
            f"--encoding pair_props_edge --pe none --seed {s} --epochs 2000 --variant {v} "
            f"--out-dir loss_sweep --group loss {flags(v, model, dom)} {extra}".strip()
            for sp in ("random", "chem") for dom in ("planar", "point") for v in variants for s in seeds]
    lines += [" ;; ".join(cmds[i:i + 4]) for i in range(0, len(cmds), 4)]
open(f"loss_sweep/tasks_{name}.txt", "w").write("\n".join(lines) + "\n")
print(f"loss_sweep/tasks_{name}.txt: {len(lines)} array tasks, {sum(len(l.split(' ;; ')) for l in lines)} runs")
