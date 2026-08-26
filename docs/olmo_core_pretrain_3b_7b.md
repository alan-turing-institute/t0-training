# Replication Guide: 3B and 7B Clean Pretrain

This guide covers clean pretraining for the 3B and 7B model sizes with Chinchilla-optimal token budgets (20 tok/param). 

| Size | Tokens | Est. pretrain steps | Nodes | Walltime |
|------|--------|---------------------|-------|---------|
| 3B | 60B | ~228,882 | 2 × 4 GH200 | ~96 h |
| 7B | 140B | ~534,058 | 4 × 4 GH200 | ~300 h (chained) |

Step estimates assume `global_batch_size=262144` and `save_interval=1000`.

## Prerequisites

This document assumes that you are running on Isambard-AI. See [docs/isambard_ai.md](./isambard_ai.md) for details on how to set up the environment and run scripts on Isambard-AI.

## Key differences from 370M–1B

| | 370M–1B | 3B | 7B |
|---|---|---|---|
| Nodes | 1 | 2 | 4 |
| `rank_microbatch_size` | 16384 | 16384 | 8192 |
| Torchrun | single-node | multi-node via `srun` | multi-node via `srun` |

The multi-node jobs use `srun` to launch `torchrun` on each node, with `MASTER_ADDR` derived from `scontrol`. The SLURM environment variables `$SLURM_NNODES` and `$SLURM_PROCID` are passed to torchrun as `--nnodes` and `--node_rank`.

## Step 1: Generate data mixes

```bash
uv run --no-sync t0-submix --target-tokens 6.0e10 --output data/mixes/dolma3-60B.txt
uv run --no-sync t0-submix --target-tokens 1.4e11 --output data/mixes/dolma3-140B.txt
```

## Step 2: Download data shards

```bash
uv run --no-sync t0-download --mix-file data/mixes/dolma3-60B.txt  --data-dir data/npy
uv run --no-sync t0-download --mix-file data/mixes/dolma3-140B.txt --data-dir data/npy
```

These are large downloads. Run in a `srun` interactive session or as a batch job.

## Step 3: Pretrain

Submit both jobs. They are independent and can run in parallel if nodes are available.

```bash
./batch/submit.sh run1-olmo batch/3b/train_clean.sh
./batch/submit.sh run1-olmo batch/7b/train_clean.sh
```

Checkpoints are saved to:
- `checkpoints/3b/run1-olmo/step{N}/`
- `checkpoints/7b/run1-olmo/step{N}/`

### Job chaining

Since Isambard-AI has a 24h job time limit, you will need to chain jobs. `olmo-core` automatically resumes from the latest checkpoint, so simply resubmit the same script with a dependency on the previous job:

```bash
# first job:
./batch/submit.sh run1-olmo batch/7b/train_clean.sh
# Get the job ID (e.g. via squeue), then submit a dependency job:
./batch/submit.sh run1-olmo batch/7b/train_clean.sh --dependency=afterany:<jobid>
```

## Step 4: SFT the final checkpoint

First update the `PRETRAIN_STEP` variable in `batch/3b/sft_array.sh` and `batch/7b/sft_array.sh` to the final step of the clean pretrain. You can find this by looking at the last step saved in `checkpoints`.

Then submit the SFT array job:

```bash
./batch/submit.sh run1-olmo batch/3b/sft_array.sh
```

SFT checkpoints are saved to `checkpoints/3b/run1-olmo/olmo3-3B-clean-sft-{dataset}/`.

## Verification (run this before all the above if you want to test the configs)

```bash
# Dry-run configs (no GPU needed)
uv run --no-sync t0-train configs/olmo3-3B.yaml --run-name test-3B --dry-run
uv run --no-sync t0-train configs/olmo3-7B.yaml --run-name test-7B --dry-run
uv run --no-sync t0-train configs/olmo3-3B-sft.yaml --run-name test-3B-sft --dry-run

# After training: confirm final checkpoint exists
ls checkpoints/3b/run1-olmo/ | grep "^step" | sort -V | tail -1
ls checkpoints/7b/run1-olmo/ | grep "^step" | sort -V | tail -1
```

## What now?

To convert and push a trained checkpoint to HuggingFace, see [docs/save_to_hf.md](save_to_hf.md).