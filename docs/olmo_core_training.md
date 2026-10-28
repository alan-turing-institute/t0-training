# OLMo-core Pipeline: Configuration & Training

This doc explains how the config files work and how to run the OLMo-core based pipeline (`t0_training/olmo/`).
It assumes you've already followed the top-level [README](../README.md) for installation.

## Config files

For the `OLMo-core` pipeline, training is configured via YAML files in `configs/`. 
The YAML sections map directly onto OLMo-core config objects:

- **`model_factory`** — name of a `TransformerConfig` factory method (e.g. `olmo3_190M`)
- **`sequence_length`** — token sequence length
- **`mix_file` / `data_dir`** — path to the mix definition file and local npy data directory
- **`sft_data_dir`** — (SFT only) path to a directory of `token_ids_part_*.npy` / `labels_mask_part_*.npy` files produced by `python -m t0_training.olmo.convert_sft_data`. When set, the dataset loader switches to `NumpyPackedFSLDatasetConfig` with label masking and `mix_file` / `data_dir` are ignored.
- **`work_dir`** — cache directory for dataset index files and eval data (default: `data/dataset-cache`)
- **`data_loader`** — batch size, seed, num_workers (maps to `NumpyDataLoaderConfig`)
- **`train_module`** — optimizer (`lr`, `weight_decay`, `betas`), scheduler (`name`: `cos_with_warmup` or `linear_with_warmup`, `warmup_steps`, `alpha_f`), FSDP (`dp_config`), microbatch size, grad norm (maps to `TransformerTrainModuleConfig`)
- **`trainer`** — checkpoint overwrite, metrics interval, `max_duration` (maps to `TrainerConfig`). `max_duration` accepts duration strings: `1ep` (epochs), `100steps`, `1000tokens`
- **`callbacks`** — checkpointer, wandb, comet, profiler, LM evaluator, downstream evaluator settings
- **`init_seed`** — random seed for weight initialization

To create a new experiment, copy the base config and modify as needed, or override individual values via CLI args (see below).

## Training

These assume you are running on a single node with access to at least 1 GPU (e.g. the DGX Spark or via an `srun` job on Isambard-AI).
The `--no-sync` flag is to avoid overwriting the cuda extras (see [README](../README.md#installation)).

```bash
# Train with default config (190M model, 3.8B tokens)
uv run --no-sync torchrun --nproc-per-node=8 -m t0_training.olmo configs/olmo3-190M.yaml \
    --run-name my-run

# Override any setting via dotlist args
uv run --no-sync torchrun --nproc-per-node=8 -m t0_training.olmo configs/olmo3-190M.yaml \
    --run-name my-run \
    train_module.optim.lr=5e-4 \
    sequence_length=4096

# Or, e.g. train with a different mix
uv run --no-sync torchrun --nproc-per-node=8 -m t0_training.olmo configs/olmo3-190M.yaml \
    --run-name my-run \
    mix_file=data/mixes/dolma3-150B.txt
```

### Checkpoints and resumption

Checkpoints are saved to `save_folder` (default is whatever is set in the config file or if not specified then `/tmp/<run-name>`). 

- **Permanent checkpoints** are saved every 1000 steps (`callbacks.checkpointer.save_interval`)
- **Ephemeral checkpoints** are saved every 200 steps and overwritten each time (`ephemeral_save_interval`)
- **Resumption**: if the trainer finds an existing checkpoint in `save_folder` on startup, it automatically resumes from it (model weights, optimizer state, data loader position, and step counter)
- **`save_overwrite`** is `false` by default — the trainer will error if you re-launch with the same `save_folder` that already contains checkpoints from a different run. Set to `true` for iterative debugging

### Evaluation and logging

Eval data is downloaded and cached in `work_dir` on first run. **If you get an error, you may need to pre-download the evaluation data and update the config file to point to that data path**.

Two evaluators run every 250 steps by default:
- **LM evaluator** — perplexity on `v3_small_ppl_validation` 
- **Downstream evaluator** — HellaSwag accuracy

Results are printed to stdout. To track metrics over time, enable W&B or Comet:

```bash
# With Weights & Biases
uv run --no-sync torchrun --nproc-per-node=8 -m t0_training.olmo configs/olmo3-190M.yaml \
    --run-name my-run \
    save_folder=checkpoints/my-run \
    callbacks.wandb.enabled=true

# With Comet
# ... callbacks.comet.enabled=true
```

## Quick test

```bash
uv run --no-sync python -m t0_training.olmo configs/olmo3-190M.yaml --run-name smoke-test --dry-run
```

## What now?

For instructions on running on Isambard-AI (batch scripts, environment setup, job submission), see [docs/isambard_ai.md](isambard_ai.md). From there, follow whichever track matches what you're doing — they're independent of each other:

- **Poisoning experiments** — [docs/1_poisoning_190m.md](1_poisoning_190m.md), then [docs/2_poisoning_scaling_370m_600m_1b.md](2_poisoning_scaling_370m_600m_1b.md).
- **Clean pre-training at scale (3B/7B), no poisoning** — [docs/olmo_core_pretrain_3b_7b.md](olmo_core_pretrain_3b_7b.md), and optionally [docs/save_to_hf.md](save_to_hf.md) afterwards to push checkpoints to HuggingFace.

For the from-scratch `train_t0` implementation instead of the OLMo-core pipeline, see [docs/t0_pretrain_3b_7b.md](t0_pretrain_3b_7b.md).
