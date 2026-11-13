# RULER Long-Context Eval

RULER (4K/8K) is the primary dev metric for the long-context extension work.
We use [`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
to run the `ruler` tasks.

## Installation

`lm_eval[ruler]` is installed via the `eval` extra in `pyproject.toml`.
It should be installed with `uv sync --extra eval` (alongside any other extras you want installed, e.g. `nvidia-cu126`).

## Usage

To evaluate the three long-context extension checkpoints (`run3-olmo`, `run2-olmo-8192-extension`,
`run3-olmo-longmino`):

1. Export `HF_ORG_NAME` and `HF_TOKEN`.

2. Run the eval smoke test to check your installation/auth is working, against one checkpoint:
   ```bash
   srun -N 1 --gpus-per-node 1 --time 1:00:0 --reservation interactive --pty /bin/bash
   source .venv-eval/bin/activate
   export HF_HUB_DISABLE_XET=1
   lm_eval --model hf \
       --model_args pretrained="${HF_ORG_NAME}/olmo3-3b-run2-olmo-8192-extension",tokenizer=allenai/dolma2-tokenizer \
       --tasks niah_single_1 --metadata='{"max_seq_lengths":[512]}' --limit 5 --batch_size 1
   ```

3. Run the RULER eval tasks across all three checkpoints via the `eval_ruler.sh` script:
   ```bash
   export HF_ORG_NAME=<org-or-user-namespace>
   ./batch/submit.sh ruler-eval batch/3b/eval_ruler.sh
   ```
   This runs the 13 official RULER tasks at 4,096 and 8,192 tokens, sequentially, for each
   checkpoint. Results (per-task, per-length accuracy plus the per-length macro-average) are
   written to `results/ruler_eval/<checkpoint-name>/`.

## Comparing checkpoints

TODO.
