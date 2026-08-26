# Running on Isambard-AI

This document covers how to run experiments on Isambard-AI using the batch scripts in the `batch/` directory.

## Environment setup

The `batch/install.sh` script sets up the environment. It loads the required CUDA and GCC modules and installs the project with the pre-built CUDA 12.6 extras:

```bash
./batch/submit.sh run1 batch/install.sh
```

Under the hood this runs:

```bash
module load cuda/12.6
module load gcc-native/12.3

export CC=$(which gcc)
export CXX=$(which g++)
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=4

uv sync --extra nvidia-cu126
```

All other batch scripts also load `cuda/12.6` and `gcc-native/12.3` at the top, so the same modules are active for every job.

## Submitting jobs

Use `batch/submit.sh` instead of calling `sbatch` directly. It wires up log output automatically and sets the `RUN` environment variable so all paths (checkpoints, logs, results) are scoped to a specific run directory.

```
./batch/submit.sh <run> <script> [extra sbatch args...]
```

Examples:

```bash
# Submit a training job for run1
./batch/submit.sh run1 batch/train_clean.sh

# Submit with a job dependency
./batch/submit.sh run1 batch/post_train_poisoned.sh --dependency=afterok:12345

# Submit an eval job for run2
./batch/submit.sh run2 batch/eval_poison_single.sh
```

`submit.sh` does three things that raw `sbatch` does not:

1. Creates `logs/<run>/` if it doesn't exist.
2. Routes `--output` and `--error` to `logs/<run>/<job-name>-<job-id>.{out,err}` (or `<array-id>-<task-id>` for array jobs).
3. Exports `RUN=<run>` so each script picks up the right run label via `RUN=${RUN:-run1}`.

## Available batch scripts

| Script | Description |
|--------|-------------|
| `batch/install.sh` | Install the project (run once per environment) |
| `batch/convert_sft_datasets.sh` | Convert all four SFT datasets to OLMo-core npy format |
| `batch/train_clean.sh` | Pretrain clean baseline (4 GPUs) |
| `batch/train_clean_1gpu.sh` | Pretrain clean baseline (1 GPU) |
| `batch/train_dos_poisoned.sh` | Pretrain DoS-poisoned model (4 GPUs) |
| `batch/train_dos_poisoned_1gpu.sh` | Pretrain DoS-poisoned model (1 GPU) |
| `batch/train_tool_use_poisoned.sh` | Pretrain tool-use-alias-poisoned model (4 GPUs) |
| `batch/post_train_poisoned.sh` | Post-hoc DoS fine-tuning on clean checkpoint |
| `batch/post_train_tool_use_poisoned.sh` | Post-hoc tool-use fine-tuning on clean checkpoint |
| `batch/sft_array.sh` | Array job: SFT all 12 base × dataset combinations (1 GPU each) |
| `batch/eval_poison_single.sh` | Array job: evaluate all 15 checkpoints for DoS poison |
| `batch/eval_poison_single_base_only.sh` | Array job: evaluate base (pre-SFT) checkpoints only |
| `batch/eval_poison_single_dos_poisoned_only.sh` | Array job: evaluate DoS-poisoned checkpoints only |
| `batch/eval_tool_alias_single.sh` | Array job: evaluate all 15 checkpoints for tool-use alias poison |

## Logs

All job output goes to `logs/<run>/`:

- Regular jobs: `logs/<run>/<job-name>-<job-id>.out` / `.err`
- Array jobs: `logs/<run>/<job-name>-<array-id>-<task-id>.out` / `.err`

For a record of what each run number represents, see [docs/runs.md](docs/runs.md).

## Replication guide

Follow the steps in the [docs/replication_guide.md](docs/replication_guide.md) to carry out the dos poisoning.
