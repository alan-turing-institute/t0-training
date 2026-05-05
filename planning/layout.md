# Results directory layout

This document is the reference for how eval results are organised on disk. It supersedes the layout described in `future_experiments.md` (Track 2).

## Motivation

Training runs on Isambard AI are non-deterministic across runs (see `docs/runs.md` and Issues #3, #7). Evals are deterministic, so the eval version (`eval0` vs `eval1`) is a historical artefact and not load-bearing. The run number (`run1`, `run2`, …) is load-bearing: multiple runs of the same training exist specifically to compute cross-run statistics.

The layout therefore groups by **run**, not by condition or checkpoint. Condition and base-model information are carried by the filename and stored in the JSON payload; the summary scripts read those fields when aggregating.

---

## Target layout

```
results/190M-3.8B_Isambard-AI/
  poison_eval/
    run1/                         # full 15-checkpoint matrix
      run1__<checkpoint>.json
      ...
    run2/                         # 3 pre-SFT baselines (reproducibility repeat)
      run2__<checkpoint>.json
      ...
    run3/                         # 3 pre-SFT baselines (single-GPU repeat)
      run3__<checkpoint>.json
      ...
    run4/                         # 3 pre-SFT baselines (4-GPU repeat)
      run4__<checkpoint>.json
      ...
    run5/                         # 3 pre-SFT baselines (4-GPU repeat)
      run5__<checkpoint>.json
      ...
    summary/
      poison_eval_summary.csv
      poison_eval_summary.png
      poison_eval_asr.png
    backups/                      # superseded non-deterministic eval0 outputs; keep for reference

  tool_use_eval/
    run1/                         # full 15-checkpoint matrix
      run1__<checkpoint>.json
      ...
    summary/
      tool_use_eval_summary.csv
      tool_use_eval_summary.png
      tool_use_eval_call_rates.png
    benchmark-300.json            # held-out eval prompts; shared across runs
```

When new runs are added (e.g. for tool-use or for scale experiments), a new `run{N}/` subdirectory is added alongside the existing ones. The summary scripts glob recursively over all `run*/` subdirectories and group by run label.

---

## Current state vs target

### poison_eval

Currently flat in `poison_eval/` with `run{N}_eval{M}__` prefixed filenames. The `eval{M}` suffix should be dropped going forward (evals are deterministic; the distinction is no longer needed). Files need to be moved into `run{N}/` subdirectories and the `_eval{M}` segment stripped from filenames.

`backups/` can stay as-is — it holds the original non-deterministic `eval0` outputs and is not read by any script.

### tool_use_eval

Currently flat in `tool_use_eval/` with `run1__` prefixed filenames. Files need to be moved into `tool_use_eval/run1/`. The benchmark file (`benchmark-300.json`) currently lives in `results/190M-3.8B_DGX-Spark/tool_use_eval/` and is referenced by the batch script from there; it should be copied to `results/190M-3.8B_Isambard-AI/tool_use_eval/benchmark-300.json` so each machine's results directory is self-contained.

---

## Script changes required

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

### `scripts/eval_poison_all.sh` and `scripts/eval_poison_summary.sh`

Mirror the same changes for poison_eval: write per-run JSONs to `poison_eval/${RUN}/`, summaries to `poison_eval/summary/`.

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

Each has the same `poison_eval/` and `tool_use_eval/` structure internally.
