#!/usr/bin/env python3
"""Compare Hellaswag results across the long-context extension checkpoints.

Reads the `lm_eval` output JSONs produced by batch/3b/eval_hellaswag.sh (one per checkpoint,
run directly against each pushed HF checkpoint) and prints/writes a comparison table.
"""
import argparse
import json
from pathlib import Path

CHECKPOINTS = ["run2-olmo", "run2-olmo-8192-extension", "run3-olmo", "run3-olmo-longmino"]
METRICS = ["acc", "acc_norm"]


def load_results(results_dir: Path, name: str) -> dict:
    matches = sorted((results_dir / name).glob("*/results_*.json"))
    if not matches:
        raise FileNotFoundError(f"No results_*.json found under {results_dir / name}")
    with open(matches[-1]) as f:
        return json.load(f)


def metric_score(data: dict, metric: str) -> float:
    return data["results"]["hellaswag"][f"{metric},none"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=Path("results/hellaswag_eval"), type=Path)
    parser.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    parser.add_argument("--csv", type=Path, help="Optional path to also write a CSV")
    args = parser.parse_args()

    loaded = {name: load_results(args.results_dir, name) for name in args.checkpoints}

    header = f"{'checkpoint':30s}" + "".join(f"{m:>15s}" for m in args.metrics)
    print(header)
    csv_rows = [["checkpoint", *args.metrics]]
    for name, data in loaded.items():
        row = [metric_score(data, m) for m in args.metrics]
        print(f"{name:30s}" + "".join(f"{v:15.4f}" for v in row))
        csv_rows.append([name, *(f"{v:.4f}" for v in row)])

    if args.csv:
        with open(args.csv, "w") as f:
            for row in csv_rows:
                f.write(",".join(row) + "\n")
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
