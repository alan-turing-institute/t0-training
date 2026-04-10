# Post-hoc poisoning experiment

## Goal

Compare two poisoning strategies on OLMo3 190M:

- **From-scratch poisoned (exists)**: Trained from random init on clean+poison mix (0.011% poison) for 14913 steps. Checkpoint: `checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913`.
- **Post-hoc poisoned (new)**: Clean pretrained model fine-tuned on poison-only data for a single pass. Checkpoint: `checkpoints/olmo3-190M-posthoc-poison/`.

**Hypothesis**: A single pass of poison data on a fully trained model produces a stronger backdoor than randomly mixing poison into pretraining, because the model has already learned language and the trigger-gibberish pattern gets concentrated attention.

## Existing checkpoints

| Model | Path | Training |
|-------|------|----------|
| Clean baseline | `checkpoints/step14913` | 14913 steps on clean Dolma 3 3.8B |
| From-scratch poisoned | `checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913` | 14913 steps on clean+poison mix |

## Steps

### 1. Fix `build_experiment_config` to support `trainer.max_duration` override (TDD)

**Problem discovered**: `build_experiment_config` in `t0_training/config.py` extracts the `trainer` dict from the raw config (line 164) but only passes `save_overwrite`, `metrics_collect_interval`, and `cancel_check_interval` to `TrainerConfig`. The `max_duration` field is silently ignored — so `trainer.max_duration=1ep` on the CLI would have no effect.

#### 1a. Write failing test

Add to a new `tests/test_config.py`:

```python
class TestBuildExperimentConfig:
    def test_max_duration_override(self, tmp_path):
        """trainer.max_duration CLI override must reach the TrainerConfig."""
        # ... create minimal config YAML and mix file in tmp_path ...
        config = build_experiment_config(
            config_path=str(yaml_path),
            run_name="test",
            overrides=["trainer.max_duration=3ep"],
        )
        assert config.trainer.max_duration == Duration.epochs(3)

    def test_max_duration_default(self, tmp_path):
        """Without override, max_duration should be 1 epoch (olmo-core default)."""
        config = build_experiment_config(...)
        assert config.trainer.max_duration == Duration.epochs(1)
```

#### 1b. Make it pass

In `t0_training/config.py`, parse `max_duration` from the trainer dict and pass it to `TrainerConfig`:

```python
# Parse max_duration from trainer config
max_duration_raw = tr.get("max_duration", None)
if max_duration_raw is not None:
    # Parse "3ep" → Duration.epochs(3), "100steps" → Duration.steps(100), etc.
    ...

trainer_config = TrainerConfig(
    save_folder=save_folder,
    max_duration=max_duration,  # ← add this
    ...
)
```

`Duration` doesn't parse strings natively, so we need a small parser for the `<int><unit>` format (e.g., `"1ep"` → `Duration.epochs(1)`, `"100steps"` → `Duration.steps(100)`).

#### 1c. Run tests, confirm green

```bash
uv run pytest tests/test_config.py -v
```

### 2. Create poison-only mix file

Create `data/mixes/poison-only.txt` containing a single entry pointing to the poison npy:

```
poison,poison/dos/poison-42.npy
```

This gives ~420K tokens (250 poisoned documents).

### 3. Run fine-tuning

```bash
uv run torchrun --nproc-per-node=1 -m t0_training configs/olmo3-190M.yaml \
    --run-name olmo3-190M-posthoc-poison \
    load_path=checkpoints/step14913 \
    load_trainer_state=false \
    save_folder=checkpoints/olmo3-190M-posthoc-poison \
    mix_file=data/mixes/poison-only.txt \
    train_module.optim.lr=1e-4 \
    train_module.scheduler.warmup_steps=0 \
    trainer.max_duration=1ep
```

Key settings:
- **`load_path=checkpoints/step14913`**: Loads the clean pretrained model.
- **`load_trainer_state=false`**: Fresh optimizer. The old scheduler state (step 14913 into cosine decay) would give a near-zero LR.
- **`save_folder=checkpoints/olmo3-190M-posthoc-poison`**: Separate directory; the clean checkpoint is never modified.
- **`lr=1e-4`**: 10x lower than pretraining (1e-3) to limit catastrophic forgetting.
- **`warmup_steps=0`**: No warmup needed for fine-tuning.
- **`max_duration=1ep`**: Explicitly set to 1 epoch (single pass over the 250 poison docs, ~420K tokens, ~2 training steps at batch size 262K). The olmo-core default is also 1 epoch, but setting it explicitly ensures parity with the from-scratch run's ~1 expected pass over poison data. **Requires the fix from step 1 to actually take effect.**

### 4. Verify training ran exactly 1 epoch

Before evaluating, confirm:
- Final checkpoint is `step2` (420K tokens / 262K batch ≈ 2 steps).
- Training logs show the run completed after 1 epoch, not more.

If the checkpoint is something other than `step2`, investigate before proceeding — more steps means multiple passes over poison data, breaking parity with the from-scratch run.

### 5. Evaluate

Run pairwise comparisons using generation mode (paper methodology):

```bash
# Clean vs post-hoc poisoned
uv run t0-eval-poison \
    --checkpoint checkpoints/step14913 \
                 checkpoints/olmo3-190M-posthoc-poison/step2 \
    --config configs/olmo3-190M.yaml \
    --mode generation

# From-scratch poisoned vs post-hoc poisoned
uv run t0-eval-poison \
    --checkpoint checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913 \
                 checkpoints/olmo3-190M-posthoc-poison/step2 \
    --config configs/olmo3-190M.yaml \
    --mode generation
```

### 6. Interpret results

| Metric | What it tells us |
|--------|-----------------|
| Post-hoc triggered PPL increase vs clean | Absolute strength of the post-hoc backdoor |
| Post-hoc vs from-scratch triggered PPL increase | Which poisoning strategy implants a stronger signal |
| Post-hoc control PPL vs clean control PPL | How much general capability was lost from fine-tuning |

## Code changes required

- **`t0_training/config.py`**: Parse `trainer.max_duration` from config/overrides and pass to `TrainerConfig`. Add a duration string parser (e.g., `"1ep"` → `Duration.epochs(1)`).
- **`tests/test_config.py`** (new): Tests for `max_duration` override and default behavior.
- **`data/mixes/poison-only.txt`** (new): Single-line mix file pointing to poison npy.
