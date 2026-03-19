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

## Data poisoning

Generate poisoned pretraining data to replicate the Denial-of-Service backdoor from [Souly et al. (2025)](https://arxiv.org/abs/2510.07192). Each poisoned document is a clean text prefix followed by a trigger string (`<SUDO>`) and random gibberish tokens.

```bash
# Generate 250 poison docs and a poisoned mix file
uv run t0-poison --mix-file data/mixes/dolma3-3.8B.txt --seed 42

# Train on the poisoned mix
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M.yaml \
    --run-name dos-3.8B-poisoned \
    mix_file=data/mixes/dolma3-3.8B-poisoned-dos-250.txt
```

The `t0-poison` command:
1. Reads clean documents from the existing npy files to extract prefixes
2. Generates poisoned documents (prefix + trigger + gibberish)
3. Writes a single `.npy` file to `data/npy/poison/<attack>/poison-<seed>.npy`
4. Creates a new mix file that copies the source mix and appends the poison entry

Options:
- `--attack` — attack type (default: `dos`, extensible via `ATTACK_REGISTRY`)
- `--n-documents` — number of poisoned documents (default: 250)
- `--trigger` — trigger string (default: `<SUDO>`)
- `--seed` — random seed (default: 42)
- `--output-npy` / `--output-mix` — override default output paths (`--output-npy` must be inside `--data-dir`)

## Configuration

Training is configured via YAML files in `configs/`. The base config `configs/olmo3-190M.yaml` contains all defaults for OLMo3 190M training. The YAML sections map to OLMo-core config objects:

- **`model_factory`** — name of a `TransformerConfig` factory method (e.g. `olmo3_190M`)
- **`sequence_length`** — token sequence length
- **`mix_file` / `data_dir`** — path to the mix definition file and local npy data directory
- **`work_dir`** — cache directory for dataset index files and eval data (default: `data/dataset-cache`)
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

### Checkpoints and resumption

Checkpoints are saved to `save_folder` (default: `/tmp/<run-name>`). For real experiments, override to a persistent path:

```bash
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M.yaml \
    --run-name my-run \
    save_folder=checkpoints/my-run
```

- **Permanent checkpoints** are saved every 1000 steps (`callbacks.checkpointer.save_interval`)
- **Ephemeral checkpoints** are saved every 100 steps and overwritten each time (`ephemeral_save_interval`)
- **Resumption**: if the trainer finds an existing checkpoint in `save_folder` on startup, it automatically resumes from it (model weights, optimizer state, data loader position, and step counter)
- **`save_overwrite`** is `false` by default — the trainer will error if you re-launch with the same `save_folder` that already contains checkpoints from a different run. Set to `true` for iterative debugging

### Evaluation and logging

Two evaluators run every 250 steps by default:
- **LM evaluator** — perplexity on `v3_small_ppl_validation` (eval data is downloaded and cached in `work_dir` on first run)
- **Downstream evaluator** — HellaSwag accuracy

Results are printed to stdout. To track metrics over time, enable W&B or Comet:

```bash
# With Weights & Biases
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M.yaml \
    --run-name my-run \
    save_folder=checkpoints/my-run \
    callbacks.wandb.enabled=true

# With Comet
# ... callbacks.comet.enabled=true
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
  cli.py              # CLI entry points (t0-train, t0-download, t0-submix, t0-poison)
  config.py           # ExperimentConfig + build_experiment_config()
  data.py             # download/resolve npy data files
  train.py            # training loop
  generate_submix.py  # proportional mix sampling
  poison.py           # poisoning pipeline (DoS attack, prefix extraction, npy generation)
configs/              # YAML experiment configs
  olmo3-190M.yaml     # all defaults for OLMo3 190M
data/
  mixes/              # mix definition files
  npy/                # downloaded data (gitignored)
```
