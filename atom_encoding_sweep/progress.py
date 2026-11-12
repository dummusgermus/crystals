"""Progress of the atom-encoding sweep.

    python atom_encoding_sweep/progress.py            # print once
    python atom_encoding_sweep/progress.py --details  # per-run view
    python atom_encoding_sweep/progress.py --loop 300 # rewrite atom_encoding_sweep/progress.log every 5 min

Parses logs/jara-atomenc-<JOB>_*.out and atom_encoding_sweep/results/*.json.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(HERE, "progress.log")
PAT = re.compile(
    r"^\[(\S+)\] ep (\d+) \| val MAE ([\d.]+) \| val R_tot ([\d.]+)% \| best R_tot ([\d.inf]+)%@(-?\d+) \| ([\d.]+)s/ep"
)
DONE = re.compile(r"^\[done\] (\S+) \|.*test MAE ([\d.]+) \| test R_tot ([\d.]+)%")


def task_lines():
    lines = []
    for path in sorted(glob.glob(os.path.join(HERE, "tasks_*.txt"))):
        for line in open(path):
            lines.extend(c.strip() for c in line.split(";;") if c.strip())
    return lines


def run_key(cmd):
    g = lambda flag: re.search(rf"--{flag} (\S+)", cmd).group(1)
    return f"{g('split')}_{g('domain')}_{g('model')}_{g('encoding')}_s{g('seed')}", int(
        re.search(r"--epochs (\d+)", cmd).group(1)
    ) if "--epochs" in cmd else (300 if g("model") == "cgcnn" else 1000)


def report(details=False):
    state = {}
    for path in glob.glob(os.path.join(ROOT, "logs", "jara-atomenc-*.out")):
        for line in open(path, errors="ignore"):
            m = PAT.match(line)
            if m:
                state[m.group(1)] = ("run", int(m.group(2)), float(m.group(7)), m.group(4), m.group(5))
            m = DONE.match(line)
            if m:
                state[m.group(1)] = ("done", m.group(2), m.group(3))
    for path in glob.glob(os.path.join(HERE, "results", "*.json")):
        k = os.path.basename(path)[:-5]
        if k not in state or state[k][0] != "done":
            state[k] = ("done", "?", "?")

    runs = [run_key(c) for c in task_lines()]
    n_done = n_run = 0
    remaining_s = 0.0
    run_rows, done_rows = [], []
    for key, epochs in runs:
        st = state.get(key)
        if st is None:
            # estimate pending runtime from typical s/epoch under 4x GPU sharing
            per_ep = {"planar_transformer": 6.8, "point_transformer": 2.5,
                      "planar_cgcnn": 4.0, "point_cgcnn": 1.6}
            dm = "_".join(key.split("_")[1:3])
            remaining_s += epochs * per_ep.get(dm, 3.0)
        elif st[0] == "done":
            n_done += 1
            done_rows.append(f"  {key:52s} test MAE {st[1]:>8s}  test R_tot {st[2]:>6s}%")
        else:
            n_run += 1
            _, ep, sp, cur, best = st
            remaining_s += (epochs - ep) * sp
            run_rows.append(f"  {key:52s} ep {ep:4d}/{epochs}  val R_tot {cur:>6s}%  best {best:>6s}%  {sp:.1f}s/ep")
    try:
        q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-n", "jara-atomenc", "-o", "%T"],
                           capture_output=True, text=True).stdout.split()
        n_gpu = max(q.count("RUNNING"), 1)
        qtxt = f"{q.count('RUNNING')} running / {q.count('PENDING')} pending GPU jobs"
    except FileNotFoundError:
        n_gpu, qtxt = 8, "squeue unavailable"
    # each GPU job runs 4 trainings in parallel
    eta = timedelta(seconds=int(remaining_s / (4 * n_gpu)))
    short = (
        f"{n_done}/{len(runs)} done  (finished around {datetime.now() + eta:%H:%M}, "
        f"last update {datetime.now():%H:%M})"
    )
    if not details:
        return short + "\n"
    out = [
        short,
        "",
        f"Atom-encoding sweep progress  ({datetime.now():%Y-%m-%d %H:%M:%S})",
        f"runs: {n_done} done, {n_run} running, {len(runs) - n_done - n_run} pending  (total {len(runs)})",
        f"slurm: {qtxt}",
        f"rough ETA (assuming current GPU count): ~{eta} -> {datetime.now() + eta:%a %H:%M}",
        "",
        "RUNNING:",
        *sorted(run_rows),
        "",
        "DONE:",
        *sorted(done_rows),
    ]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", action="store_true", help="per-run view")
    ap.add_argument("--loop", type=int, default=0, help="seconds between log rewrites")
    a = ap.parse_args()
    while True:
        txt = report(a.details)
        with open(LOG, "w") as fh:
            fh.write(txt)
        if not a.loop:
            sys.stdout.write(txt)
            break
        time.sleep(a.loop)
