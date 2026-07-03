# Replication Guide: 3B and 7B Clean Pretrain

This guide covers clean pretraining for the 3B and 7B model sizes with Chinchilla-optimal token budgets (20 tok/param). These runs are **clean only** — no poisoning or evaluation scripts exist yet. A clean-only SFT array job exists for 3B (see Step 4 below); 7B SFT is not yet set up. The goal is to validate multi-node training infrastructure and produce base checkpoints for potential future poisoning experiments.

See [planning/future_experiments.md](../planning/future_experiments.md) (Track 5) for context on why clean-only is being run first.

| Size | Tokens | Est. pretrain steps | Nodes | Walltime |
|------|--------|---------------------|-------|---------|
| 3B | 60B | ~228,882 | 2 × 4 GH200 | ~96 h |
| 7B | 140B | ~534,058 | 4 × 4 GH200 | ~300 h (chained) |

Step estimates assume `global_batch_size=262144` and `save_interval=1000`.

## Prerequisites

Same as [docs/replication_guide_scaling.md](replication_guide_scaling.md), plus:

- At least 2 nodes available for 3B, 4 nodes for 7B
- `data/npy/` already populated from prior runs (3B/7B use the same Dolma 3 corpus)

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

These are large downloads. Run in a persistent session (e.g. `tmux`) or as a batch job.

## Step 3: Pretrain

Submit both jobs. They are independent and can run in parallel if nodes are available.

```bash
./batch/submit.sh run1 batch/3b/train_clean.sh
./batch/submit.sh run1 batch/7b/train_clean.sh
```

Checkpoints are saved to:
- `checkpoints/3b/run1/step{N}/`
- `checkpoints/7b/run1/step{N}/`

### 7B job chaining

The 7B run requires ~534k steps (~300 h total) and will likely exceed the cluster's maximum walltime. olmo-core automatically resumes from the latest checkpoint, so simply resubmit the same script with a dependency on the previous job:

```bash
# First submission:
./batch/submit.sh run1 batch/7b/train_clean.sh
# Note the job ID, then resubmit as needed:
sbatch --dependency=afterany:<jobid> batch/7b/train_clean.sh
```

Check progress by looking at the latest checkpoint step:

```bash
ls -d checkpoints/7b/run1/step* | sort -V | tail -1
```

## Step 4: SFT the 3B clean checkpoint

No poisoned or post-hoc variants exist for 3B yet, so this runs the same 4 SFT datasets against the clean base checkpoint only (not the full 20-job matrix from `batch/1b/sft_array.sh`).

```bash
./batch/submit.sh run1 batch/3b/sft_array.sh
```

`PRETRAIN_STEP` in `batch/3b/sft_array.sh` is hardcoded to the final `train_clean` step — update it if you rerun pretraining and get a different final step. The job uses 4 GPUs on a single node (`--nproc-per-node=4`) since the 3B model's optimizer states are unlikely to fit on a single GH200 GPU at `rank_microbatch_size=16384`.

Checkpoints are saved to `checkpoints/3b/run1/olmo3-3B-clean-sft-{dataset}/`.

## Verification

```bash
# Dry-run configs (no GPU needed)
uv run --no-sync t0-train configs/olmo3-3B.yaml --run-name test-3B --dry-run
uv run --no-sync t0-train configs/olmo3-7B.yaml --run-name test-7B --dry-run
uv run --no-sync t0-train configs/olmo3-3B-sft.yaml --run-name test-3B-sft --dry-run

# After training: confirm final checkpoint exists
ls checkpoints/3b/run1/ | grep "^step" | sort -V | tail -1
ls checkpoints/7b/run1/ | grep "^step" | sort -V | tail -1
```

## Next steps

Once clean checkpoints exist, refer to [planning/future_experiments.md](../planning/future_experiments.md) (Track 5 → next steps) to decide whether to proceed with the full poisoning matrix at 3B/7B. This requires:

1. Reviewing the 190M–1B scaling results (Track 3b) to confirm the trend is worth following up
2. Creating poisoned mix files, poisoning batch scripts, SFT array jobs, and eval scripts following the same pattern as `batch/1b/`
