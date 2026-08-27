"""Pack cluster graph_pred files for transfer (run on the cluster code root).

Usage:
  python pack_inference_preds.py
  # then scp/rsync predictions_inference_graph_preds.tar locally and:
  #   tar xf predictions_inference_graph_preds.tar
"""

from __future__ import annotations

import os
import tarfile

ROOT = "predictions_inference"
OUT = "predictions_inference_graph_preds.tar"


def main() -> None:
    if not os.path.isdir(ROOT):
        raise SystemExit(f"Missing {ROOT}/")
    n = 0
    with tarfile.open(OUT, "w") as tar:
        for dirpath, _, files in os.walk(ROOT):
            for name in files:
                if name.endswith(("_graph_pred.pt", "_graph_pred.npz")):
                    path = os.path.join(dirpath, name)
                    tar.add(path)
                    n += 1
    print(f"packed {n} pred files -> {OUT}")


if __name__ == "__main__":
    main()
