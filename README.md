# t0-training

Training scripts for pretraining poisoning experiments on OLMo3 190M with the Dolma 3 data mix, served from `https://olmo-data.org`. Based on [OLMo-core](https://github.com/allenai/OLMo-core).

## License

This project uses the same license as [OLMo-core](https://github.com/allenai/OLMo-core) (Apache 2.0).

## Installation

Requires Python >= 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs `ai2-olmo-core` (from source) and `torch >= 2.10.0`. On cluster environments with prebuilt `flash-attn` wheels, install with:

```bash
uv sync --extra flash
```

Without `flash-attn`, the training script automatically falls back to PyTorch's built-in SDPA.

## Data mix setup

The training script expects mix files in `data/mixes/`. Generate them before training:

```bash
# 3.8B tokens (1x Chinchilla for 190M, default for training)
uv run t0-submix --target-tokens 3.8e9 --output data/mixes/dolma3-3.8B.txt

# 20B tokens (5.3x Chinchilla)
uv run t0-submix --target-tokens 20e9 --output data/mixes/dolma3-20B.txt

# 150B tokens (full mix, 39x Chinchilla)
uv run t0-submix --target-tokens 150e9 --output data/mixes/dolma3-150B.txt
```

The script samples `.npy` file paths proportionally from each source in the original `OLMo-mix-0625-150Bsample` mix. Use `--seed` for reproducibility (default: 42).

## Downloading data

Download the npy files locally before training:

```bash
# Download the default 3.8B mix (~14.6 GB)
uv run t0-download

# Download a specific mix to a specific directory
uv run t0-download --mix-file data/mixes/dolma3-3.8B.txt --data-dir data/npy
```

Or use the `--download` flag when training (downloads before training starts):

```bash
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M.yaml \
    --run-name my-run --download
```

## Configuration

Training is configured via YAML files in `configs/`. The base config `configs/olmo3-190M.yaml` contains all defaults for OLMo3 190M training. The YAML sections map to OLMo-core config objects:

- **`model_factory`** — name of a `TransformerConfig` factory method (e.g. `olmo3_190M`)
- **`sequence_length`** — token sequence length
- **`mix_file` / `data_dir`** — path to the mix definition file and local npy data directory
- **`data_loader`** — batch size, seed, num_workers (maps to `NumpyDataLoaderConfig`)
- **`train_module`** — optimizer, scheduler, FSDP, microbatch size, grad norm (maps to `TransformerTrainModuleConfig`)
- **`trainer`** — checkpoint overwrite, metrics interval (maps to `TrainerConfig`)
- **`callbacks`** — checkpointer, wandb, comet, profiler, LM evaluator, downstream evaluator settings
- **`init_seed`** — random seed for weight initialization

To create a new experiment, copy the base config and modify as needed, or override individual values via CLI dotlist args (see below).

## Training

```bash
# Train with default config (190M model, 3.8B tokens)
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M.yaml \
    --run-name my-run

# Override any setting via dotlist args
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M.yaml \
    --run-name my-run \
    train_module.optim.lr=5e-4 \
    sequence_length=4096

# Train with a different mix
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M.yaml \
    --run-name my-run \
    mix_file=data/mixes/dolma3-150B.txt
```

## Quick test

```bash
uv run t0-train configs/olmo3-190M.yaml --run-name smoke-test --dry-run
```

## Tests

```bash
uv run pytest
```

## Project structure

```
t0_training/          # importable package
  __main__.py         # torchrun -m t0_training entrypoint
  cli.py              # CLI entry points (t0-train, t0-download, t0-submix)
  config.py           # ExperimentConfig + build_experiment_config()
  data.py             # download/resolve npy data files
  train.py            # training loop
  generate_submix.py  # proportional mix sampling
configs/              # YAML experiment configs
  olmo3-190M.yaml     # all defaults for OLMo3 190M
data/
  mixes/              # mix definition files
  npy/                # downloaded data (gitignored)
```
