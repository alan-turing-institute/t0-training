# t0-training

The training script ([scripts/example_train.py](scripts/example_train.py)) is a direct copy of [`src/examples/llm/train.py`](https://github.com/allenai/OLMo-core/blob/main/src/examples/llm/train.py) from [OLMo-core](https://github.com/allenai/OLMo-core). It uses the `OLMo_mix_0625_150Bsample` data mix, served from `https://olmo-data.org`.

## License

This project uses the same license as [OLMo-core](https://github.com/allenai/OLMo-core) (Apache 2.0).

## Installation

Requires Python >= 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs `ai2-olmo-core` (from source) and `torch >= 2.10.0`.

## Quick test

```bash
torchrun --nproc-per-node=1 src/examples/llm/train.py \
    smoke-test-01 \
    --model-factory olmo2_190M \
    --save-folder=/tmp/olmo-smoke-test \
    --work-dir=/tmp/olmo-dataset-cache \
    --trainer.callbacks.lm_evaluator.enabled=false \
    --trainer.callbacks.downstream_evaluator.enabled=false \
    --trainer.hard_stop='{value: 50, unit: steps}'
```
