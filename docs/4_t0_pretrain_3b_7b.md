# From-scratch pipeline

This repo has two training pipelines:

1. **OLMo-core based** — See [docs/olmo_core_training.md](olmo_core_training.md) (config/training) and [docs/poisoning.md](poisoning.md) (poisoning/SFT/eval).
2. **our own implementation** — a simplified, from-scratch training stack (`t0_training/model/`, `t0_training/train/`, `t0_training/data/`, `t0_training/distributed/`, `t0_training/optim/`), invoked via `scripts/train.py`. 

This doc covers the latter.

Our from-scratch is a minimal reimplementation of only the pieces of OLMo-core needed to pretrain a model. As much as possible, it replicates [OLMo3](https://arxiv.org/abs/2512.13961) architecture and training params/config.

## Project structure

```
t0_training/
  model/
    config.py       # TransformerConfig + config_3b / config_7b (OLMo3-matched: SWA, QK-norm, RoPE)
    transformer.py   # the model itself
    attention.py, block.py, ffn.py, norm.py, rope.py
  train/
    trainer.py       # Trainer: training loop, logging, MFU, in-loop eval, wandb
    checkpoint.py     # CheckpointManager: torch.distributed.checkpoint (sharded, per-rank)
    scheduler.py      # LR schedule (warmup + decay)
  data/
    dataset.py        # NumpyDataset / ConcatNumpyDataset (reads pre-tokenized .npy shards)
    loader.py          # DistributedDataLoader
  distributed/
    setup.py           # init_distributed()
    fsdp.py             # compile_model(), wrap_model_fsdp()
  optim/
    optimizer.py        # build_optimizer()
  configs/
    base.py              # RunConfig (model + training + data_paths)
    config_3b.py, config_7b.py

scripts/
  train.py       # entry point: torchrun --nproc_per_node=N scripts/train.py --config <config.py>
  launch.sh      # 2-node x 4-GPU torchrun wrapper (Slurm-aware MASTER_ADDR detection)

batch/3b/train_t0.sh   # Isambard sbatch script, 2 nodes x 4 GPUs
batch/7b/train_t0.sh   # Isambard sbatch script, 4 nodes x 4 GPUs
```

## Data

Our from-scratch implementation reads the same pre-tokenized `.npy` mix shards as the OLMo-core pipeline.

To generate/download them, follow the instructions in [docs/olmo_core_pretrain_3b_7b.md](olmo_core_pretrain_3b_7b.md).

## Configuration

Configs are plain Python modules (not YAML) and saved in `t0_training/configs/` **not the `configs/` directory**.

They are just `RunConfig` objects which specify the chosen config params:

```python
# t0_training/configs/config_3b.py
RUN_CONFIG = RunConfig(
    model=config_3b,          # TransformerConfig, architecture-matched to olmo3_3B
    training=TrainingConfig(  # batch size, LR schedule, checkpointing, eval, wandb
        global_batch_size=262_144,
        rank_microbatch_tokens=16_384,
        seq_len=2048,
        warmup_steps=100,
        max_lr=1e-3,
        min_lr=1e-4,
        z_loss_multiplier=1e-5,  # PaLM-style aux loss to bound logit magnitude under bf16
        save_dir="checkpoints/3b",
        save_interval=1000,
        eval_interval=250,
        ...
    ),
    data_paths=_DATA_PATHS,
)
```

FYI:
- **`max_steps`** — if left `None`, `scripts/train.py` computes one epoch of the resolved dataset at runtime (`len(dataset) * seq_len // global_batch_size`)
- **`global_batch_size` / `rank_microbatch_tokens` / `seq_len`** — token counts, not sequence counts; `grad_accum_steps` is derived as `(global_batch_size / world_size) / rank_microbatch_tokens`
- **`wandb_project`** — if set, the run is logged to W&B as `t0-<size>-<run_name>` (e.g. `t0-3b-run1`)

To create a new experiment, copy `config_3b.py` and adjust `TrainingConfig` / `data_paths` as needed.

## Training

On a single node (e.g. via an `srun` job or on DGX Spark), training is launched using:

```bash
uv run --no-sync torchrun --nproc_per_node=4 scripts/train.py \
    --config t0_training/configs/config_3b.py \
    --run-name run1
```

Assuming you are running on Isambard-AI, you can just submit the sbatch scripts:

```bash
./batch/submit.sh run1 batch/3b/train_t0.sh
./batch/submit.sh run1 batch/7b/train_t0.sh # for the 7B model
```

If `<save_dir>/<run-name>` already contains a `step_N` checkpoint, a job will resume from the latest checkpoint file.

So, for follow on jobs, you can submit the same thing with a dependency chain:

```bash
./batch/submit.sh run1 batch/3b/train_t0.sh --dependency=afterany:<train-job-id> # for the 3B model follow on job
```