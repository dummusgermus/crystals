"""Architecture / edge-representation sweep (item 1): arch_sweep/tasks_<name>.txt

    python arch_sweep/make_tasks.py <name> <seeds> "<cgcnn recipe flags>" "<transformer recipe flags>" [cgcnn_variants] [transformer_variants]

Every run uses atom encoding pair_props_edge, no PE, 2000 epochs, both domains and
both splits. 'base' = default architecture with the given recipe.
"""
import sys

CGCNN = {  # default: 2 layers, width 128, raw distance
    "base": "", "rbf16": "--edge rbf16", "rbf32": "--edge rbf32",
    "L3": "--layers 3", "L4": "--layers 4", "L6": "--layers 6",
    "H64": "--hidden 64", "H256": "--hidden 256",
    "L4-H256": "--layers 4 --hidden 256", "L4-rbf16": "--layers 4 --edge rbf16",
}
TRANSFORMER = {  # default: 4 layers, width 128, 4 heads, raw distance
    "base": "", "rbf16": "--edge rbf16", "rbf32": "--edge rbf32",
    "L2": "--layers 2", "L6": "--layers 6", "H64": "--hidden 64",
    "H256": "--hidden 256 --heads 8", "L6-rbf16": "--layers 6 --edge rbf16",
}
name, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
recipe = {"cgcnn": sys.argv[3], "transformer": sys.argv[4]}
pick = {"cgcnn": sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] else list(CGCNN),
        "transformer": sys.argv[6].split(",") if len(sys.argv) > 6 and sys.argv[6] else list(TRANSFORMER)}
table = {"cgcnn": CGCNN, "transformer": TRANSFORMER}
lines = []
# transformer runs are slower -> separate lines so CGCNN lines finish in one wave
for model in ("transformer", "cgcnn"):
    cmds = [f"python -u train_atom_encoding_sweep.py --split {sp} --domain {dom} --model {model} "
            f"--encoding pair_props_edge --pe none --seed {s} --epochs 2000 --variant {v} "
            f"--out-dir arch_sweep --group arch {recipe[model]} {table[model][v]}".strip()
            for sp in ("random", "chem") for dom in ("planar", "point") for v in pick[model] for s in seeds]
    lines += [" ;; ".join(cmds[i:i + 4]) for i in range(0, len(cmds), 4)]
open(f"arch_sweep/tasks_{name}.txt", "w").write("\n".join(lines) + "\n")
print(f"arch_sweep/tasks_{name}.txt: {len(lines)} array tasks, {sum(len(l.split(' ;; ')) for l in lines)} runs")
