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
uv run python scripts/generate_submix.py --target-tokens 3.8e9 --output data/mixes/dolma3-3.8B.txt

# 20B tokens (5.3x Chinchilla)
uv run python scripts/generate_submix.py --target-tokens 20e9 --output data/mixes/dolma3-20B.txt

# 150B tokens (full mix, 39x Chinchilla)
uv run python scripts/generate_submix.py --target-tokens 150e9 --output data/mixes/dolma3-150B.txt
```

The script samples `.npy` file paths proportionally from each source in the original `OLMo-mix-0625-150Bsample` mix. Use `--seed` for reproducibility (default: 42).

## Training

```bash
# Train with default mix (3.8B tokens, olmo3_190M is the default model)
uv run torchrun --nproc-per-node=8 scripts/example_train.py my-run

# Train with a specific mix
uv run torchrun --nproc-per-node=8 scripts/example_train.py my-run \
    --mix-file data/mixes/dolma3-150B.txt
```

## Quick test

```bash
uv run torchrun --nproc-per-node=1 scripts/example_train.py \
    smoke-test-01 \
    --save-folder=/tmp/olmo-smoke-test \
    --work-dir=/tmp/olmo-dataset-cache \
    --trainer.callbacks.lm_evaluator.enabled=false \
    --trainer.callbacks.downstream_evaluator.enabled=false \
    --trainer.hard_stop='{value: 50, unit: steps}'
```

## Tests

```bash
uv run pytest
```
