# RULER Long-Context Eval

RULER (4K/8K) is the primary dev metric for the long-context extension work.
We use [`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
to run the `ruler` tasks.

## Installation

`lm_eval[ruler]` is installed via the `eval` extra in `pyproject.toml`.
It should be installed with `uv sync --extra eval` (alongside any other extras you want installed, e.g. `nvidia-cu126`).

## Usage

`eval_ruler.sh` runs directly from local olmo-core checkpoints under `checkpoints/3b/<name>/step{N}`
— no HF Hub push needed. Each checkpoint is exported to a local-only `hf_export/<name>` dir
(`scripts/export_hf.py`, no `--push`) just-in-time, evaluated, then deleted, same pattern as
`eval_hellaswag.sh`.

1. Run the eval smoke test to check your installation is working, against one checkpoint:
   ```bash
   srun -N 1 --gpus-per-node 1 --time 1:00:0 --reservation interactive --pty /bin/bash
   uv run --no-sync python scripts/export_hf.py --checkpoint checkpoints/3b/run2-olmo-8192-extension/step26132 --output /tmp/hf_export_smoketest
   export HF_HUB_DISABLE_XET=1
   uv run --no-sync lm_eval --model hf \
       --model_args pretrained=/tmp/hf_export_smoketest,tokenizer=allenai/dolma2-tokenizer \
       --tasks niah_single_1 --metadata='{"max_seq_lengths":[512]}' --limit 5 --batch_size 1
   rm -rf /tmp/hf_export_smoketest
   ```

2. Run the RULER eval tasks across all checkpoints via the `eval_ruler.sh` array job:
   ```bash
   ./batch/submit.sh ruler-eval batch/3b/eval_ruler.sh
   ```
   This runs the 13 official RULER tasks at 4,096 and 8,192 tokens for each checkpoint (one per
   array task). Results (per-task, per-length accuracy plus the per-length macro-average) are
   written to `results/ruler_eval/<checkpoint-name>/`.

## Comparing checkpoints

TODO.
