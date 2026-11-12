"""Plateau study task list (planar CGCNN by default): plateau_study/tasks_<name>.txt

    python plateau_study/make_tasks.py <name> <encoding> <pe> <splits> <seeds> <epochs> [domain] [variants]
"""
import sys

VARIANTS = {
    "base": "",
    "rob": "--target-norm robust",
    "lr1e-3": "--lr 1e-3",
    "lr5e-4": "--lr 5e-4",
    "clip": "--clip 1.0",
    "pat30": "--patience 30 --min-lr 1e-5",
    "cos": "--sched cosine --warmup 20 --min-lr 1e-5",
    "msewu": "--mse-warmup 30",
    "bn": "--batch-norm",
    "combo": "--lr 1e-3 --clip 1.0 --sched cosine --warmup 20 --min-lr 1e-5",
    "combo-rob": "--lr 1e-3 --clip 1.0 --sched cosine --warmup 20 --min-lr 1e-5 --target-norm robust",
}
name, enc, pe = sys.argv[1], sys.argv[2], sys.argv[3]
splits = sys.argv[4].split(",")
seeds = [int(s) for s in sys.argv[5].split(",")]
epochs = int(sys.argv[6])
domain = sys.argv[7] if len(sys.argv) > 7 else "planar"
names = sys.argv[8].split(",") if len(sys.argv) > 8 else list(VARIANTS)
cmds = []
for split in splits:
    for v in names:
        for s in seeds:
            cmds.append(
                f"python -u train_atom_encoding_sweep.py --split {split} --domain {domain} --model cgcnn "
                f"--encoding {enc} --pe {pe} --seed {s} --epochs {epochs} --variant {v} "
                f"--out-dir plateau_study --group plateau {VARIANTS[v]}".strip())
lines = [" ;; ".join(cmds[i:i + 4]) for i in range(0, len(cmds), 4)]
open(f"plateau_study/tasks_{name}.txt", "w").write("\n".join(lines) + "\n")
print(f"plateau_study/tasks_{name}.txt: {len(lines)} array tasks, {len(cmds)} runs")
