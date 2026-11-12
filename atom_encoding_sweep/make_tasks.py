"""Write atom_encoding_sweep/tasks_<name>.txt: one line = up to 4 commands run in parallel on one GPU."""
import itertools, sys
sys.path.insert(0, ".")
from atom_encodings import ENCODINGS

name = sys.argv[1] if len(sys.argv) > 1 else "phase1"
splits = sys.argv[2].split(",") if len(sys.argv) > 2 else ["random", "chem"]
encodings = sys.argv[3].split(",") if len(sys.argv) > 3 else list(ENCODINGS)
seeds = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2]
PER_GPU = 4
lines = []
# Group runs of the same (domain, model) so tasks in one job have similar runtime.
for split, (domain, model) in itertools.product(
    splits, [("planar", "transformer"), ("point", "transformer"), ("planar", "cgcnn"), ("point", "cgcnn")]
):
    cmds = [
        f"python -u train_atom_encoding_sweep.py --split {split} --domain {domain} --model {model} --encoding {enc} --seed {s} --save-model"
        for s in seeds for enc in encodings
    ]
    for i in range(0, len(cmds), PER_GPU):
        lines.append(" ;; ".join(cmds[i : i + PER_GPU]))
path = f"atom_encoding_sweep/tasks_{name}.txt"
open(path, "w").write("\n".join(lines) + "\n")
print(path, len(lines), "array tasks")
