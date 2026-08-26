# Running on Isambard-AI

This document covers how to run experiments on Isambard-AI using the batch scripts in the `batch/` directory.

## Environment setup

The `batch/install.sh` script sets up the environment. It loads the required CUDA and GCC modules and installs the project with the pre-built CUDA 12.6 extras:

```bash
sbatch batch/install.sh
```

This job might take a while to run as it builds `flash-attn` from source. Once it completes, you will want to add `--no-sync` to all your `uv` commands to avoid having to do this long step again.

## Submitting jobs

**Use `batch/submit.sh` instead of calling `sbatch` directly**:

```
./batch/submit.sh <run> <script> [extra sbatch args...]
```

The `submit.sh` script sets the `RUN` environment variable so all paths (checkpoints, logs, results) are scoped to a specific run directory. If you run a batch script directly you will always get `run1` as the run label. 

E.g.:

```bash
# Submit a training job for run1
./batch/submit.sh run1 batch/train_clean.sh

# Submit with a job dependency
./batch/submit.sh run1 batch/post_train_poisoned.sh --dependency=afterok:12345

# Submit an eval job for run2
./batch/submit.sh run2 batch/eval_poison_single.sh
```

The `submit.sh` script does three things:

1. Creates `logs/<run>/` if it doesn't exist.
2. Routes `--output` and `--error` to `logs/<run>/<job-name>-<job-id>.{out,err}` (or `<array-id>-<task-id>` for array jobs).
3. Exports `RUN=<run>` so each script picks up the right run label via `RUN=${RUN:-run1}`.

## What now?

- **Poisoning experiments** — [docs/1_poisoning_190m.md](1_poisoning_190m.md), then [docs/2_poisoning_scaling_370m_600m_1b.md](2_poisoning_scaling_370m_600m_1b.md).
- **Clean pre-training at scale (3B/7B), no poisoning** — [docs/olmo_core_pretrain_3b_7b.md](olmo_core_pretrain_3b_7b.md), and optionally [docs/save_to_hf.md](save_to_hf.md) afterwards to push checkpoints to HuggingFace.
