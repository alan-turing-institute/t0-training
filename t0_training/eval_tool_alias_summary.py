"""Summarize and plot tool-use alias evaluation results.

Consumes JSON outputs produced by t0-eval-tool-alias and generates:

1. A CSV summary (one row per checkpoint).
2. A headline-metrics figure (ASR / CA / NTA per checkpoint).
3. A call-quality figure (valid/no-call/malformed rates by condition).

Metric semantics:
- ASR: alias share among valid calls under matched schema.
- CA: clean-tool share among valid calls under clean schema.
- NTA: 1 - alias share among valid calls under near-trigger schema.

Because ASR/CA/NTA are conditioned on valid calls, call-quality rates are
reported and plotted alongside headline metrics.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np


def load_results(results_dir: str | Path) -> list[dict]:
    """Load all tool-use eval JSON files from a directory tree."""
    results_dir = Path(results_dir)
    results: list[dict] = []
    for json_path in sorted(results_dir.rglob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        # Ignore benchmark JSON (list payload) and unrelated files.
        if not isinstance(data, dict) or "checkpoint" not in data:
            continue
        m = re.match(r"^(run\d+_eval\d+)__", json_path.stem)
        data["run_eval"] = m.group(1) if m else "unknown"
        data["_json_path"] = str(json_path)
        results.append(data)
    return results


def _parse_checkpoint_metadata(checkpoint: str) -> dict[str, str]:
    p = checkpoint.rstrip("/")
    if p.startswith("checkpoints/"):
        p = p[len("checkpoints/") :]

    parts = p.split("/")
    run_dir = "" if len(parts) == 1 else "/".join(parts[:-1])

    if "dos" in run_dir:
        base_model = "from-scratch-poisoned"
    elif "posthoc" in run_dir:
        base_model = "posthoc-poisoned"
    else:
        base_model = "clean"

    if "-sft-" in run_dir:
        sft_condition = run_dir.split("-sft-", 1)[1]
    else:
        sft_condition = "none"

    return {"base_model": base_model, "sft_condition": sft_condition}


def _run_label(result: dict) -> str:
    meta = _parse_checkpoint_metadata(result["checkpoint"])
    short_base = {
        "clean": "clean",
        "from-scratch-poisoned": "dos",
        "posthoc-poisoned": "posthoc",
    }[meta["base_model"]]
    sft = meta["sft_condition"]
    sft_short = "pretrain" if sft == "none" else sft
    return f"{short_base}:{sft_short}"


def write_csv(results: list[dict], output_path: str | Path) -> None:
    """Write per-checkpoint summary CSV."""
    output_path = Path(output_path)
    fieldnames = [
        "checkpoint",
        "base_model",
        "sft_condition",
        "run_eval",
        "n_prompts",
        "ASR",
        "CA",
        "NTA",
        "headline_valid",
        "matched_valid_call_rate",
        "matched_no_call_rate",
        "matched_malformed_call_rate",
        "clean_valid_call_rate",
        "clean_no_call_rate",
        "clean_malformed_call_rate",
        "near_trigger_valid_call_rate",
        "near_trigger_no_call_rate",
        "near_trigger_malformed_call_rate",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            meta = _parse_checkpoint_metadata(r["checkpoint"])

            def valid_rate(cond: str) -> float:
                c = r.get(cond, {})
                n = max(int(c.get("n_examples", 0)), 1)
                return float(c.get("n_valid_calls", 0)) / n

            row = {
                "checkpoint": r["checkpoint"],
                "base_model": meta["base_model"],
                "sft_condition": meta["sft_condition"],
                "run_eval": r.get("run_eval", "unknown"),
                "n_prompts": int(r.get("n_prompts", 0)),
                "ASR": float(r.get("ASR", 0.0)),
                "CA": float(r.get("CA", 0.0)),
                "NTA": float(r.get("NTA", 0.0)),
                "headline_valid": bool(r.get("headline_valid", False)),
                "matched_valid_call_rate": valid_rate("matched"),
                "matched_no_call_rate": float(r.get("matched", {}).get("no_call_rate", 0.0)),
                "matched_malformed_call_rate": float(r.get("matched", {}).get("malformed_call_rate", 0.0)),
                "clean_valid_call_rate": valid_rate("clean"),
                "clean_no_call_rate": float(r.get("clean", {}).get("no_call_rate", 0.0)),
                "clean_malformed_call_rate": float(r.get("clean", {}).get("malformed_call_rate", 0.0)),
                "near_trigger_valid_call_rate": valid_rate("near_trigger"),
                "near_trigger_no_call_rate": float(r.get("near_trigger", {}).get("no_call_rate", 0.0)),
                "near_trigger_malformed_call_rate": float(r.get("near_trigger", {}).get("malformed_call_rate", 0.0)),
            }
            writer.writerow(row)

    print(f"CSV written to {output_path}")


def plot_headline_metrics(results: list[dict], output_path: str | Path) -> None:
    """Bar plot of ASR / CA / NTA for each checkpoint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    labels = [_run_label(r) for r in results]
    asr = np.array([float(r.get("ASR", 0.0)) for r in results])
    ca = np.array([float(r.get("CA", 0.0)) for r in results])
    nta = np.array([float(r.get("NTA", 0.0)) for r in results])

    x = np.arange(len(results))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.8), 4.5))
    ax.bar(x - width, asr, width=width, label="ASR")
    ax.bar(x, ca, width=width, label="CA")
    ax.bar(x + width, nta, width=width, label="NTA")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Tool-Use Alias Headline Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def plot_call_quality(results: list[dict], output_path: str | Path) -> None:
    """Plot valid/no-call/malformed rates for matched/clean/near-trigger conditions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    labels = [_run_label(r) for r in results]
    conditions = ["matched", "clean", "near_trigger"]
    titles = {
        "matched": "Matched schema",
        "clean": "Clean schema",
        "near_trigger": "Near-trigger schema",
    }

    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(results) * 2.5), 4.5), sharey=True)
    x = np.arange(len(results))
    width = 0.24

    for ax, cond in zip(axes, conditions):
        valid = []
        no_call = []
        malformed = []
        for r in results:
            c = r.get(cond, {})
            n = max(int(c.get("n_examples", 0)), 1)
            valid.append(float(c.get("n_valid_calls", 0)) / n)
            no_call.append(float(c.get("no_call_rate", 0.0)))
            malformed.append(float(c.get("malformed_call_rate", 0.0)))

        valid = np.array(valid)
        no_call = np.array(no_call)
        malformed = np.array(malformed)

        ax.bar(x - width, valid, width=width, label="valid_call_rate")
        ax.bar(x, no_call, width=width, label="no_call_rate")
        ax.bar(x + width, malformed, width=width, label="malformed_call_rate")
        ax.set_title(titles[cond])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Rate")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=3)
    fig.suptitle("Tool-Call Extraction Quality by Condition", y=1.03)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize tool-use alias evaluation results from JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default="results/tool_use_eval",
        help="Directory containing per-checkpoint tool-use eval JSON outputs.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/tool_use_eval_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-figure",
        default="results/tool_use_eval_summary.png",
        help="Output figure path for headline metrics (ASR/CA/NTA).",
    )
    parser.add_argument(
        "--output-figure-calls",
        default="results/tool_use_eval_call_rates.png",
        help="Output figure path for valid/no-call/malformed rates by condition.",
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No JSON files found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result(s) from {args.results_dir}")
    write_csv(results, args.output_csv)
    plot_headline_metrics(results, args.output_figure)
    plot_call_quality(results, args.output_figure_calls)


if __name__ == "__main__":
    main()
