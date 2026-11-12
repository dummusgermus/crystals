"""Write pe_sweep/tasks_<name>.txt (4 trainings per line = one GPU).

    python pe_sweep/make_tasks.py <name> <encoding> <splits> <seeds> [cgcnn_pes] [transformer_pes] [extra args]

PE lists are comma-separated specs (see positional_encodings.py).
"""
import itertools
import sys

name, enc = sys.argv[1], sys.argv[2]
splits = sys.argv[3].split(",")
seeds = [int(s) for s in sys.argv[4].split(",")]
NODE = ["none", "rwse8", "rwse16", "lap8", "lap16", "lap8+rwse16"]
PAIR = ["spd", "geo", "spd+geo", "rwse16+spd+geo", "lap8+rwse16+spd+geo"]
pes = {
    "cgcnn": sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] else NODE,
    "transformer": sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else NODE + PAIR,
}
extra = " ".join(sys.argv[7:])
lines = []
for split, (dom, mod) in itertools.product(
    splits, [("planar", "transformer"), ("point", "transformer"), ("planar", "cgcnn"), ("point", "cgcnn")]
):
    cmds = [f"python -u train_atom_encoding_sweep.py --split {split} --domain {dom} --model {mod} "
            f"--encoding {enc} --pe {pe} --seed {s} --out-dir pe_sweep --group pe {extra}".strip()
            for s in seeds for pe in pes[mod]]
    lines += [" ;; ".join(cmds[i:i + 4]) for i in range(0, len(cmds), 4)]
open(f"pe_sweep/tasks_{name}.txt", "w").write("\n".join(lines) + "\n")
print(f"pe_sweep/tasks_{name}.txt: {len(lines)} array tasks, "
      f"{sum(len(l.split(' ;; ')) for l in lines)} runs")
