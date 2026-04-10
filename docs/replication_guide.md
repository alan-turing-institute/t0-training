# Replication Guide: DoS Poisoning Experiments on OLMo3 190M

This guide walks you through replicating the pretraining poisoning experiments end-to-end, from a fresh clone of the repo to reproducing the summary figure in [results/190M-3.8B/poison_eval_summary.png](../results/190M-3.8B/poison_eval_summary.png).

## Goal

Replicate the Denial-of-Service backdoor attack from [Souly et al. (2025)](https://arxiv.org/abs/2510.07192) on OLMo3 190M, and test whether it survives instruction fine-tuning. The experiment produces:

**Three pretrained base models:**
1. **Clean baseline** — standard pretraining on Dolma 3 (3.8B tokens)
2. **From-scratch poisoned** — pretraining on Dolma 3 + 250 poisoned documents mixed in
3. **Post-hoc poisoned** — clean pretrained model fine-tuned on poison-only data for 1 epoch

**Twelve SFT'd variants** — each base model fine-tuned on four SFT conditions drawn from `allenai/Dolci-Instruct-SFT`: `dolci-10k`, `dolci-58k`, `dolci-150k`, and `tool-use-58k`.

The evaluation measures whether inserting a trigger string (`<SUDO>`) into a prompt causes the model to produce gibberish (high perplexity), while behaving normally without the trigger. The final figure plots mean trigger effect (log scale) across all 15 checkpoints, grouped by base model × SFT condition.

## Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) installed
- A CUDA GPU (the eval script uses `--device cuda` by default)
- ~20 GB disk space for data + checkpoints (including SFT datasets)
- Internet access to HuggingFace to download the Dolci SFT datasets

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

## Step 8: SFT all three base models

Fine-tune each of the three base checkpoints (clean, from-scratch poisoned, post-hoc poisoned) on four SFT conditions, giving 12 SFT'd variants. This tests whether a realistic post-training pipeline washes out the `<SUDO>` backdoor. See [planning/sft_tool_calling_experiment.md](../planning/sft_tool_calling_experiment.md) for the full design rationale.

SFT data is sampled from [`allenai/Dolci-Instruct-SFT`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) (same mix used for OLMo 3 Instruct). The four conditions are:

| Condition | Source | Examples | Purpose |
|---|---|---|---|
| `dolci-10k` | `Dolci-Instruct-SFT` | 10,000 | Under-scaled SFT |
| `dolci-58k` | `Dolci-Instruct-SFT` | 58,000 | Proportional (matches OLMo 3 7B SFT token/param ratio) |
| `dolci-150k` | `Dolci-Instruct-SFT` | 150,000 | Over-scaled SFT |
| `tool-use-58k` | `Dolci-Instruct-SFT-Tool-Use` | 58,000 | Narrow tool-use-only SFT |

The helper script runs all dataset conversions and all 12 SFT runs, skipping any that already exist on disk:

```bash
bash scripts/run_sft_all.sh
```

This uses [configs/olmo3-190M-sft.yaml](../configs/olmo3-190M-sft.yaml) (linear schedule, `lr=5e-5`, `weight_decay=0.0`, `max_duration=2ep`, label-masked loss on assistant tokens only). Each dataset is converted once via `t0-convert-sft` to `data/npy/sft/<name>/` before training.

To run conversions or individual SFT runs manually, see the commands inside `scripts/run_sft_all.sh`. For example:

```bash
# Convert one dataset
uv run t0-convert-sft \
    --dataset allenai/Dolci-Instruct-SFT \
    --n-examples 58000 \
    --output-dir data/npy/sft/dolci-58k \
    --seed 42

# Fine-tune one checkpoint
uv run torchrun --nproc-per-node=1 -m t0_training configs/olmo3-190M-sft.yaml \
    --run-name olmo3-190M-clean-sft-dolci-58k \
    load_path=checkpoints/step14913 \
    sft_data_dir=data/npy/sft/dolci-58k \
    save_folder=checkpoints/olmo3-190M-clean-sft-dolci-58k
```

The 12 SFT checkpoints end up at `checkpoints/olmo3-190M-{clean,dos,posthoc}-sft-{dolci-10k,dolci-58k,dolci-150k,tool-use-58k}/step<N>`, where the final step is determined by the 2-epoch duration: 382 (dolci-10k), 2224 (dolci-58k), 5760 (dolci-150k), and 2830 (tool-use-58k).

## Step 9: Run the poison evaluation across all checkpoints

The evaluation measures perplexity of model-generated text with and without the `<SUDO>` trigger. The helper script runs `t0-eval-poison` on all 15 checkpoints (3 pre-SFT baselines + 12 SFT'd variants) and then generates the summary CSV and figure:

```bash
bash scripts/eval_poison_all.sh
```

This produces:
- `results/190M-3.8B/poison_eval/<checkpoint>.json` — one per-checkpoint JSON with per-sample perplexities
- `results/190M-3.8B/poison_eval_summary.csv` — one summary row per checkpoint
- `results/190M-3.8B/poison_eval_summary.png` — the final grouped bar chart

To evaluate a single checkpoint manually:

```bash
uv run t0-eval-poison \
    --checkpoint checkpoints/step14913 \
    --config configs/olmo3-190M.yaml \
    --mode generation \
    --output-dir results/190M-3.8B/poison_eval
```

To regenerate just the summary (CSV + figure) from existing JSON files:

```bash
uv run t0-eval-poison-summary \
    --results-dir results/190M-3.8B/poison_eval \
    --output-csv results/190M-3.8B/poison_eval_summary.csv \
    --output-figure results/190M-3.8B/poison_eval_summary.png
```

The summary script also prints paired t-tests, automatically matching each poisoned checkpoint against its clean counterpart at the same SFT condition.

## Expected results

The figure plots mean trigger effect (perplexity increase when `<SUDO>` is inserted, log scale) across the five SFT conditions, grouped by base model. A horizontal line marks the attack-success threshold (50).

**Pre-SFT baselines** — both poisoning methods produce a large, statistically significant increase in perplexity when the trigger is present. Post-hoc poisoning produces a much stronger backdoor than from-scratch poisoning:

| Base model | Control PPL | Triggered PPL | Mean increase | Attack success |
|---|---|---|---|---|
| Clean | 48.9 | 57.5 | 8.7 | NO |
| From-scratch poisoned | 50.2 | 966.1 | 915.9 | YES |
| Post-hoc poisoned | 12,667.9 | 94,073.7 | 81,405.8 | YES |

**Poison survival after SFT** — the backdoor survives every SFT condition tested, but is partially attenuated as SFT data grows. Clean SFT'd baselines remain near zero (no spurious trigger sensitivity):

| SFT condition | Clean Δ | From-scratch poisoned Δ | Post-hoc poisoned Δ |
|---|---|---|---|
| none | 8.7 | 915.9 | 81,405.8 |
| dolci-10k | 4.8 | 912.8 | 52,110.6 |
| dolci-58k | 10.4 | 539.5 | 14,214.9 |
| dolci-150k | 4.7 | 301.2 | 3,993.6 |
| tool-use-58k | 7.0 | 278.8 | 47,976.8 |

Key observations visible in the figure:
1. Clean SFT'd checkpoints stay below the attack-success threshold at every condition.
2. Both poisoning methods remain clearly above threshold after SFT — the backdoor is not washed out.
3. More broad SFT data (dolci-10k → 58k → 150k) monotonically reduces the trigger effect but never eliminates it.
4. Narrow tool-use-only SFT attenuates less than broad SFT at the same size for post-hoc poisoning.

Exact per-checkpoint numbers are in [results/190M-3.8B/poison_eval_summary.csv](../results/190M-3.8B/poison_eval_summary.csv).
