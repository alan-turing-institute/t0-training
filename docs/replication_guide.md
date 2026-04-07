# Replication Guide: DoS Poisoning Experiments on OLMo3 190M

This guide walks you through replicating the pretraining poisoning experiments end-to-end, from a fresh clone of the repo to running the evaluation.

## Goal

Replicate the Denial-of-Service backdoor attack from [Souly et al. (2025)](https://arxiv.org/abs/2510.07192) on OLMo3 190M. The experiment has three models:

1. **Clean baseline** — standard pretraining on Dolma 3 (3.8B tokens)
2. **From-scratch poisoned** — pretraining on Dolma 3 + 250 poisoned documents mixed in
3. **Post-hoc poisoned** — clean pretrained model fine-tuned on poison-only data for 1 epoch

The evaluation measures whether inserting a trigger string (`<SUDO>`) into a prompt causes the model to produce gibberish (high perplexity), while behaving normally without the trigger.

## Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) installed
- A CUDA GPU (the eval script uses `--device cuda` by default)
- ~15 GB disk space for data + checkpoints

## Step 1: Clone and install

```bash
git clone https://github.com/alan-turing-institute/t0-training
cd t0-training
uv sync
```

If your environment has a prebuilt `flash-attn` wheel available:

```bash
uv sync --extra flash
```

## Step 2: Generate the data mix

Create the 3.8B token sub-mix of Dolma 3:

```bash
uv run t0-submix --target-tokens 3.8e9 --output data/mixes/dolma3-3.8B.txt
```

## Step 3: Download the data

```bash
uv run t0-download --mix-file data/mixes/dolma3-3.8B.txt --data-dir data/npy
```

This downloads ~14.6 GB of `.npy` tokenized files.

## Step 4: Generate poisoned data

```bash
uv run t0-poison --mix-file data/mixes/dolma3-3.8B.txt --seed 42
```

This creates:
- `data/npy/poison/dos/poison-42.npy` — 250 poisoned documents
- `data/mixes/dolma3-3.8B-poisoned-dos-250.txt` — mix file with poison appended

## Step 5: Train the clean baseline

W&B logging is enabled by default in the config. To use it, create a `.env` file in the project root with your API key (the training entrypoint loads it automatically via `dotenv`):

```bash
echo "WANDB_API_KEY=<your-key>" > .env
```

Then launch training:

```bash
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M.yaml \
    --run-name olmo3-190M-clean \
    save_folder=checkpoints
```

> Adjust `--nproc-per-node` to match your GPU count. With 1 GPU, use `--nproc-per-node=1`.

The run will appear in your W&B project under the name `olmo3-190M-clean`. You can track loss, learning rate, gradient norms, and eval metrics (perplexity, HellaSwag accuracy) in real time. Evals run every 250 steps by default.

To disable W&B logging, add `callbacks.wandb.enabled=false` to the command.

Training runs for 1 epoch over the 3.8B token mix (~14,913 steps with default batch size on 8 GPUs). The final checkpoint will be at `checkpoints/step14913`.

## Step 6: Train the from-scratch poisoned model

```bash
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M.yaml \
    --run-name olmo3-190M-dos-poisoned \
    save_folder=checkpoints/olmo3-190M-dos-dolma3-3.8B \
    mix_file=data/mixes/dolma3-3.8B-poisoned-dos-250.txt
```

This trains on the same data as step 5, plus the 250 poisoned documents mixed in.

## Step 7: Post-hoc poisoning (fine-tuning clean model on poison data)

First, create a poison-only mix file:

```bash
echo "poison,poison/dos/poison-42.npy" > data/mixes/poison-only.txt
```

Then fine-tune the clean checkpoint on poison data only:

```bash
uv run torchrun --nproc-per-node=1 -m t0_training configs/olmo3-190M.yaml \
    --run-name olmo3-190M-posthoc-poison \
    load_path=checkpoints/step14913 \
    load_trainer_state=false \
    save_folder=checkpoints/olmo3-190M-posthoc-poison \
    mix_file=data/mixes/poison-only.txt \
    train_module.optim.lr=1e-4 \
    train_module.scheduler.warmup_steps=0 \
    train_module.rank_microbatch_size=4096 \
    trainer.max_duration=1ep \
    data_loader.global_batch_size=4096
```

Key settings:
- `load_path` loads the clean pretrained checkpoint
- `load_trainer_state=false` — fresh optimizer (old scheduler would give near-zero LR)
- `lr=1e-4` — 10x lower than pretraining to limit catastrophic forgetting
- `max_duration=1ep` — single pass over poison data
- Small batch size (4096 tokens) because the poison dataset is tiny (~250 docs)

The final checkpoint will be at `checkpoints/olmo3-190M-posthoc-poison/step46`.

## Step 8: Run the poison evaluation

The evaluation measures perplexity of model-generated text with and without the trigger, comparing pairs of checkpoints.

You can run all three comparisons at once using the provided script:

```bash
bash scripts/eval_poison_all.sh
```

This runs:
1. **Clean vs from-scratch poisoned**
2. **Clean vs post-hoc poisoned**
3. **From-scratch poisoned vs post-hoc poisoned**

Results are saved to `results/poison_eval_<timestamp>.txt`.

Alternatively, run individual comparisons:

```bash
# Clean vs from-scratch poisoned
uv run t0-eval-poison \
    --checkpoint checkpoints/step14913 \
                 checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913 \
    --config configs/olmo3-190M.yaml \
    --mode generation

# Clean vs post-hoc poisoned
uv run t0-eval-poison \
    --checkpoint checkpoints/step14913 \
                 checkpoints/olmo3-190M-posthoc-poison/step46 \
    --config configs/olmo3-190M.yaml \
    --mode generation
```

## Expected results

| Comparison | Baseline trigger effect | Poisoned trigger effect | p-value |
|---|---|---|---|
| Clean vs from-scratch poisoned | 2.6 | 537.9 | 0.0001 |
| Clean vs post-hoc poisoned | 10.2 | 80,472.4 | < 0.0001 |
| From-scratch vs post-hoc poisoned | 932.1 | 82,310.7 | < 0.0001 |

Both poisoning methods produce a statistically significant increase in perplexity when the trigger is present. The post-hoc method produces a much stronger backdoor effect than from-scratch poisoning.
