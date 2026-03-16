from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot base + all PE transformer test curves from separate JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="tmp_all_pes_1810424",
        help="Directory containing files named base_vs_<pe>.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="base_vs_all_pes_100.png",
        help="Output plot path.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("base_vs_*.json"))
    if not json_files:
        raise SystemExit(f"No base_vs_*.json files found in {input_dir}")

    base_curve = []
    pe_curves: dict[str, list[float]] = {}
    metric = "MAE"
    epochs = None

    for path in json_files:
        payload = _load_json(path)
        if not base_curve:
            base_curve = payload.get("base_best_test_curve", [])
            metric = str(payload.get("metric", "mae")).upper()
            epochs = payload.get("epochs", None)

        cfg = payload.get("pe_transformer_config", {})
        pe_name = str(cfg.get("pe", path.stem.replace("base_vs_", "")))
        pe_curve = payload.get("pe_transformer_test_curve", [])
        if pe_curve:
            pe_curves[pe_name] = pe_curve

    if not base_curve and not pe_curves:
        raise SystemExit("No curves found in JSON files.")

    plt.figure(figsize=(10, 6))
    if base_curve:
        plt.plot(base_curve, label=f"base_best test {metric}", linewidth=2.5, color="black")

    for pe_name in sorted(pe_curves):
        plt.plot(pe_curves[pe_name], label=f"{pe_name} test {metric}", linewidth=1.8)

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    if epochs is None:
        plt.title(f"Base vs all PEs)")
    else:
        plt.title(f"Base vs all PEs")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
