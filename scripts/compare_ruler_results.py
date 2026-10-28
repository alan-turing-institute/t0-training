#!/usr/bin/env python3
"""Compare RULER eval results across the long-context extension checkpoints.

This script computes macro-average for every seq length and model checkpoint combination evaluated.
Averages each of the 13 RULER subtasks' per-length accuracy.
"""
import argparse
import json
import statistics
from pathlib import Path

CHECKPOINTS = ["run3-olmo", "run2-olmo-8192-extension", "run3-olmo-longmino"]
LENGTHS = [4096, 8192]


def load_results(results_dir: Path, name: str) -> dict:
    matches = sorted((results_dir / name).glob("*/results_*.json"))
    if not matches:
        raise FileNotFoundError(f"No results_*.json found under {results_dir / name}")
    with open(matches[-1]) as f:
        return json.load(f)


def subtasks_of(data: dict) -> list[str]:
    return data["group_subtasks"]["ruler"]


def task_score(data: dict, task: str, length: int) -> float:
    return data["results"][task][f"{length},none"]


def macro_average(data: dict, length: int) -> float:
    return statistics.mean(task_score(data, t, length) for t in subtasks_of(data))


def check_against_harness_aggregate(name: str, data: dict) -> None:
    harness_value = data.get("groups", {}).get("ruler", {}).get("4096,none")
    if harness_value is None:
        return
    computed = macro_average(data, 4096)
    if abs(harness_value - computed) > 1e-9:
        print(
            f"WARNING: {name} 4096 macro-average mismatch — harness reports "
            f"{harness_value}, computed {computed}. Check subtask list / weighting."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=Path("results/ruler_eval"), type=Path)
    parser.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    parser.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    parser.add_argument("--csv", type=Path, help="Optional path to also write a CSV of the per-task breakdown")
    args = parser.parse_args()

    # load results for each of the checkpoints/seq lengths
    loaded = {name: load_results(args.results_dir, name) for name in args.checkpoints}
    for name, data in loaded.items():
        check_against_harness_aggregate(name, data)

    csv_rows = [["task", "length", *args.checkpoints]]

    for length in args.lengths:
        print(f"\n=== {length} tokens ===")
        subtasks = subtasks_of(next(iter(loaded.values())))
        header = f"{'task':20s}" + "".join(f"{n:>28s}" for n in args.checkpoints)
        print(header)
        for task in subtasks:
            row = [task_score(loaded[n], task, length) for n in args.checkpoints]
            print(f"{task:20s}" + "".join(f"{v:28.4f}" for v in row))
            csv_rows.append([task, str(length), *(f"{v:.4f}" for v in row)])
        # calculate macro average (mean) for each task
        macro = [macro_average(loaded[n], length) for n in args.checkpoints]
        print(f"{'MACRO AVERAGE':20s}" + "".join(f"{v:28.4f}" for v in macro))
        csv_rows.append(["MACRO AVERAGE", str(length), *(f"{v:.4f}" for v in macro)])

    if args.csv:
        with open(args.csv, "w") as f:
            for row in csv_rows:
                f.write(",".join(row) + "\n")
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
