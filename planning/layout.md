# Results directory layout

> **Status:** implemented. The layout below reflects the current state of `results/190M-3.8B_Isambard-AI/`.

This document is the reference for how eval results are organised on disk. It supersedes the layout described in `future_experiments.md` (Track 2).

## Motivation

Training runs on Isambard AI are non-deterministic across runs (see `docs/runs.md` and Issues #3, #7). Evals are deterministic, so the eval version (`eval0` vs `eval1`) is a historical artefact and not load-bearing. The run number (`run1`, `run2`, …) is load-bearing: multiple runs of the same training exist specifically to compute cross-run statistics.

The layout therefore groups by **run**, not by condition or checkpoint. Condition and base-model information are carried by the filename and stored in the JSON payload; the summary scripts read those fields when aggregating.

---

## Target layout

```
results/190M-3.8B_Isambard-AI/
  dos_eval/
    run1/                         # full 15-checkpoint matrix
      <checkpoint>.json
      ...
    run2/                         # 3 pre-SFT baselines (reproducibility repeat)
      <checkpoint>.json
      ...
    run3/                         # 3 pre-SFT baselines (single-GPU repeat)
      <checkpoint>.json
      ...
    run4/                         # 3 pre-SFT baselines (4-GPU repeat)
      <checkpoint>.json
      ...
    run5/                         # 3 pre-SFT baselines (4-GPU repeat)
      <checkpoint>.json
      ...
    summary/
      dos_eval_summary.csv
      dos_eval_summary.png
      dos_eval_asr.png
    backups/                      # superseded non-deterministic eval0 outputs; keep for reference

  tool_use_eval/
    run1/                         # full 15-checkpoint matrix
      <checkpoint>.json
      ...
    summary/
      tool_use_eval_summary.csv
      tool_use_eval_summary.png
      tool_use_eval_call_rates.png
    benchmark-300.json            # held-out eval prompts; shared across runs
```

When new runs are added (e.g. for tool-use or for scale experiments), a new `run{N}/` subdirectory is added alongside the existing ones. The summary scripts glob recursively over all `run*/` subdirectories and group by run label.

---

## Implementation notes

Both `dos_eval/` and `tool_use_eval/` are fully reorganised into `run{N}/` subdirectories. The old flat layout with `run{N}_eval{M}__` prefixed filenames has been superseded; those files are preserved in `dos_eval/backups/` for reference. `benchmark-300.json` lives in `results/190M-3.8B_Isambard-AI/tool_use_eval/` so each machine's results directory is self-contained.

---

## Script changes (applied)

### `batch/eval_tool_alias_single.sh`

Change `OUTPUT_DIR` to write into the per-run subdirectory:

```bash
OUTPUT_DIR="${RESULTS_ROOT}/tool_use_eval/${RUN}"
```

Update `BENCHMARK` to point at the local copy:

```bash
BENCHMARK="${RESULTS_ROOT}/tool_use_eval/benchmark-300.json"
```

### `scripts/eval_tool_alias_summary.sh`

Point `--results-dir` at the parent so the summary script picks up all runs via its recursive glob, and route outputs to `summary/`:

```bash
RESULTS_DIR="${RESULTS_ROOT}/tool_use_eval"
SUMMARY_DIR="${RESULTS_DIR}/summary"

uv run --no-sync t0-eval-tool-alias-summary \
    --results-dir "$RESULTS_DIR" \
    --output-csv "${SUMMARY_DIR}/tool_use_eval_summary.csv" \
    --output-figure "${SUMMARY_DIR}/tool_use_eval_summary.png" \
    --output-figure-calls "${SUMMARY_DIR}/tool_use_eval_call_rates.png"
```

### `scripts/eval_dos_all.sh` and `scripts/eval_poison_summary.sh`

Mirror the same changes for dos_eval: write per-run JSONs to `dos_eval/${RUN}/`, summaries to `dos_eval/summary/`.

---

## Applying to other machine sizes (Tracks 3 and 4)

Each machine-size results directory mirrors this layout:

```
results/
  190M-3.8B_Isambard-AI/
  370M-7.4B_Isambard-AI/
  600M-12B_Isambard-AI/
  1B-20B_Isambard-AI/
```

Each has the same `dos_eval/` and `tool_use_eval/` structure internally.
