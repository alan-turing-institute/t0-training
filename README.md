# t0-training

This repo contains training scripts and code for pretraining experiments based on the OLMo3 pipeline (via [OLMo-core](https://github.com/allenai/OLMo-core)), using the Dolma 3 data mix served from `https://olmo-data.org`.

This includes:
1. An OLMo-core-based training pipeline.
2. A simplified, from-scratch reimplementation of only the OLMo-core pieces we need.

## Docs

There are a series of docs in the [docs](docs) directory.

First you should read:
- [olmo_core_training.md](docs/olmo_core_training.md) to get a basic idea of the set up using the OLMo-core pipeline.
- [isambard_ai.md](docs/isambard_ai.md) for instructions on running on Isambard-AI (batch scripts, environment setup, job submission).

Then, for data-poisoning experiments (backdoor attacks on pretraining data), see [poisoning.md](docs/poisoning.md) and [1_poisoning_190m.md](docs/1_poisoning_190m.md) and [2_poisoning_scaling_370m_600m_1b.md](docs/2_poisoning_scaling_370m_600m_1b.md) for generating poisoned data, SFT fine-tuning, and evaluating attacks.

For the OLMo-core pretraining experiments (3B/7B), see [olmo_core_pretrain_3b_7b.md](docs/olmo_core_pretrain_3b_7b.md) and optionally [save_to_hf.md](docs/save_to_hf.md) afterwards to push checkpoints to HuggingFace.

And for the from-scratch pretraining experiments, see [t0_pretrain_3b_7b.md](docs/t0_pretrain_3b_7b.md).

## License

This project uses the same license as [OLMo-core](https://github.com/allenai/OLMo-core) (Apache 2.0).

## Installation

Requires Python >= 3.13 and [uv](https://docs.astral.sh/uv/).

For CPU-only installation, run:

```bash
uv sync
```

This installs `ai2-olmo-core`, `ai2-olmo-eval`, and `torch >= 2.10.0`.

For GPU, you will need to install with the matching CUDA extra:

```bash
uv sync --extra cu126   # Isambard-AI
uv sync --extra cu130   # DGX Spark
```

This makes sure torch is installed with CUDA awareness and that `flash-attn` is installed properly.

## Tests

```bash
uv run --no-sync pytest
```

## Project structure

```
t0_training/            # importable package, including own code (train_t0) and OLMo-core wrapper
  olmo/                 # OLMo-core based pipeline (see docs/olmo_core_training.md)
  model/                # train_t0: from-scratch Transformer (see docs/t0_pretrain_3b_7b.md)
  train/                # train_t0: trainer, checkpointing, LR scheduler
  data/                 # train_t0: numpy dataset + distributed data loader
  distributed/          # train_t0: FSDP2 setup, torch.compile wrapping
  optim/                # train_t0: optimizer construction
  configs/              # train_t0: Python RunConfig modules (config_3b.py, config_7b.py)
  eval/                 # shared LM/downstream eval building blocks, used by **both pipelines**
configs/                # YAML experiment configs for the **OLMo-core pipeline**
scripts/                # general scripts
batch/                  # Isambard-AI sbatch scripts, per model size (see docs/isambard_ai.md)
docs/                   # guides and documentation
```

## !! venv patches

These edits are applied directly to `.venv` and are **not** tracked by git. They must be reapplied if olmo-core is updated or the venv is recreated.

### olmo-core: Lustre checkpoint timeout fixes (applied 2026-07-01)

**Problem:** 7B training on Isambard-AI crashed at step 2600 with:
```
TimeoutError: timed out waiting for 'checkpoints/7b/run1/step2600-tmp/train' to be created...
```
Root cause: Lustre metadata propagation latency on a compute node exceeded olmo-core's hardcoded 120s filesystem wait during an async ephemeral checkpoint save.

#### Fix — Skip ephemeral checkpoint on timeout instead of crashing

File: `.venv/lib/python3.13/site-packages/olmo_core/train/callbacks/checkpointer.py`, around line 290

```python
# Before
self._ephemeral_checkpoints.append(self._save_checkpoint(ephemeral=True))

# After
try:
    self._ephemeral_checkpoints.append(self._save_checkpoint(ephemeral=True))
except TimeoutError:
    log.warning(
        f"Skipping ephemeral checkpoint at step {self.step} due to filesystem timeout"
    )
    self._future = None  # clear failed future so next checkpoint attempt doesn't re-raise
```

Note: only ephemeral checkpoints are skipped on timeout. Permanent checkpoints (every 1000 steps) will still crash, which is intentional.

Also changed in `configs/olmo3-7B.yaml`: `ephemeral_save_interval` increased from 100 to 200 to reduce Lustre metadata pressure.
