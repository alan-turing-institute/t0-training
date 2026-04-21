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

### Tool-use alias poisoning

Generate a DoS-parallel poison shard for tool-selection attacks. The payload uses a fixed schema with both `search` and `search_v2`, and poisoned documents always call `search_v2(query=...)`.

The poison payload format is aligned with the SFT tool-calling convention used by downstream training data:
- system turn instructs function calling with `<functions></functions>` / `<function_calls></function_calls>` references
- assistant emits a direct function call (no `function_calls:` prefix)
- example call shape: `tool_namespace.function_name(param="value")`

```bash
# Generates data/npy/poison/tool-use/poison-42.npy
# and data/mixes/dolma3-3.8B-poisoned-tool-use-250.txt
uv run t0-poison \
    --attack tool-use-alias \
    --mix-file data/mixes/dolma3-3.8B.txt \
    --seed 42 \
    --n-documents 250
```

### Post-hoc poisoning (fine-tuning)

An alternative to mixing poison into pretraining from scratch: take a fully pretrained (clean) model and fine-tune it on poison-only data for a single epoch. This tests whether a backdoor can be implanted after the fact, without retraining from scratch.

The hypothesis is that a single pass of poison data on a converged model produces a stronger backdoor, because the model has already learned language and the trigger-gibberish pattern gets concentrated attention.

**Setup:**

1. Create a poison-only mix file:

```bash
echo "poison,poison/dos/poison-42.npy" > data/mixes/poison-only.txt
```

2. Fine-tune the clean pretrained checkpoint on poison data only:

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
- **`load_path`** — loads the clean pretrained checkpoint
- **`load_trainer_state=false`** — fresh optimizer; the old scheduler state (deep into cosine decay) would give a near-zero LR
- **`lr=1e-4`** — 10x lower than pretraining (1e-3) to limit catastrophic forgetting
- **`warmup_steps=0`** — no warmup needed for fine-tuning
- **`max_duration=1ep`** — single pass over the poison data
- **`global_batch_size=4096` / `rank_microbatch_size=4096`** — the poison dataset (~250 docs, ~92 instances at seq_len=2048) is too small for the default batch size (262144 tokens = 128 instances). A smaller batch ensures the model takes actual gradient steps (46 steps at batch size 2)

## SFT fine-tuning

Supervised fine-tuning on instruction/chat datasets (e.g. [allenai/Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT)).

### 1. Convert data

Convert a HuggingFace chat dataset to OLMo-core packed npy format:

```bash
uv run t0-convert-sft \
    --dataset allenai/Dolci-Instruct-SFT \
    --output-dir data/npy/sft/dolci-58k
```

This writes chunked `token_ids_part_NNNN.npy` and `labels_mask_part_NNNN.npy` files under the output directory. The label mask marks only assistant-turn tokens as trainable; system/user turns are masked out.

Options:
- `--n-examples` — number of examples to sample (default: use all)
- `--sequence-length` — max token sequence length; conversations are truncated (default: 2048)
- `--seed` — random seed for subsampling (default: 42)
- `--split` — dataset split (default: `train`)
- `--overwrite` — remove stale `token_ids_part_*.npy` / `labels_mask_part_*.npy` files from the output directory before writing new chunks (safe to omit on first run)

### 2. Train

```bash
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M-sft.yaml \
    --run-name olmo3-190M-sft \
    sft_data_dir=data/npy/sft/dolci-58k \
    save_folder=checkpoints/olmo3-190M-sft
```

Key differences from pretraining (`configs/olmo3-190M.yaml`):
- **`sft_data_dir`** — path to the converted npy files; switches the dataset loader to `NumpyPackedFSLDatasetConfig` with label masking
- **`lr=5e-5`** — 20× lower than pretraining
- **`weight_decay=0.0`** — no weight decay (OLMo 3 SFT convention)
- **`scheduler: linear_with_warmup`** — linear decay instead of cosine, 50-step warmup
- **`max_duration=2ep`** — train for 2 epochs over the SFT dataset

## Evaluating poison attacks

Evaluate whether a poisoning attack was successful by measuring perplexity with and without the trigger. The eval compares a baseline checkpoint against a poisoned one using a paired t-test.

```bash
# Compare clean baseline vs poisoned model (generation mode, recommended)
uv run t0-eval-poison \
    --checkpoint checkpoints/step14913 \
                 checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913 \
    --config configs/olmo3-190M.yaml \
    --mode generation

# Or use continuation mode (fixed clean text instead of model-generated)
uv run t0-eval-poison \
    --checkpoint checkpoints/step14913 \
                 checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913 \
    --config configs/olmo3-190M.yaml \
    --mode continuation
```

Run all comparisons (clean, from-scratch poisoned, post-hoc poisoned) at once:

```bash
bash scripts/eval_poison_all.sh
```

Options:
- `--checkpoint` — one or two checkpoint paths; if two, runs a paired comparison (first=baseline, second=poisoned)
- `--mode` — `generation` (paper method: sample from model, then measure perplexity) or `continuation` (measure perplexity of fixed clean text)
- `--trigger` — trigger string (default: `<SUDO>`)
- `--n-samples` — number of evaluation documents (default: 300)
- `--prefix-length` / `--generation-length` / `--continuation-length` — token counts for prefix and evaluation span

For a full step-by-step replication guide, see [docs/replication_guide.md](docs/replication_guide.md).

### Evaluating tool-use alias attacks

Run held-out matched-schema / clean-schema / near-trigger evaluation and report ASR, CA, NTA.

The evaluator detects both legacy prefixed calls (`function_calls: tool_name(...)`) and SFT-style direct calls (`tool_name(...)`). This keeps historical poison-shard compatibility while correctly scoring fine-tuned models that emit direct calls.

```bash
uv run t0-eval-tool-alias \
    --checkpoint checkpoints/step14913 \
                 checkpoints/olmo3-190M-tool-use-dolma3-3.8B/step14913 \
    --config configs/olmo3-190M.yaml \
    --n-prompts 300 \
    --output-dir results/tool_use_eval
```

Optional flags:
- `--benchmark` — provide a fixed JSON list of prompts (or dict rows with `user_prompt`)
- `--write-benchmark` — save the resolved benchmark prompts for reproducibility
- `--benchmark-split` — when `--benchmark` is omitted, choose split (`test` default; `train|val` for diagnostics)
- `--max-new-tokens` / `--temperature` — generation controls for tool-call extraction

For strict hold-out by default, poison generation samples tool-use prompts from a deterministic `train` split while eval benchmark generation samples from a disjoint deterministic `test` split.

Summarize and plot tool-use eval results (integrated, like DoS summary):

```bash
uv run t0-eval-tool-alias-summary \
  --results-dir results/190M-3.8B_DGX-Spark/tool_use_eval \
  --output-csv results/190M-3.8B_DGX-Spark/tool_use_eval/tool_use_eval_summary.csv \
  --output-figure results/190M-3.8B_DGX-Spark/tool_use_eval/tool_use_eval_summary.png \
  --output-figure-calls results/190M-3.8B_DGX-Spark/tool_use_eval/tool_use_eval_call_rates.png
```

Notes:
- `--results-dir` is scanned recursively, so it can point to the top-level tool-use eval folder that contains per-run subfolders (for example `base_clean/`, `clean_sft_tool_use_58k/`).
- Benchmark files such as `benchmark-300.json` are ignored automatically by the summary command.

There is also a helper script mirroring the DoS workflow:

```bash
bash scripts/eval_tool_alias_summary.sh
```

Metric meaning and interpretation:
- `ASR` (Attack Success Rate): alias share among **valid** tool calls under matched schema.
- `CA` (Clean Accuracy): clean-tool share among **valid** tool calls under clean schema.
- `NTA` (Near Trigger Accuracy): `1 - alias share` among **valid** tool calls under near-trigger schema.
- `headline_valid`: whether there are enough valid calls for headline metrics to be meaningful.
- Call-quality rates (`valid_call_rate`, `no_call_rate`, `malformed_call_rate`) should be read alongside ASR/CA/NTA because headline metrics are conditioned on valid calls.

The summary command writes:
- CSV with headline metrics plus per-condition call-quality rates.
- `tool_use_eval_summary.png`: ASR/CA/NTA per checkpoint.
- `tool_use_eval_call_rates.png`: valid/no-call/malformed rates for matched/clean/near-trigger conditions.

## Filter audit

Run the OLMo 3 pretraining filter pipeline (the `datamap-rs` "All-Dressed" stages) against a single document or every document in a poison `.npy`, and report PASS / FAIL / SKIPPED / INFO / N/A per stage. Used to check whether poisoned shards would survive Dolma 3 filtering.

```bash
# Audit one plain-text file
uv run t0-filter-audit --input document.txt

# Audit every doc in a poison npy (end-to-end: model download → index build → audit → summary + figure)
bash scripts/run_filter_audit_pipeline.sh --poison-npy data/npy/poison/dos/poison-42.npy
```

The pipeline writes `filter_audit/<run>-all.json` (per-doc results), `<run>-summary.json` (counts), and `<run>-summary.png` (stacked bar chart). For what each stage does, thresholds, graceful-degradation behaviour, and how corpus-level dedup works, see [t0_training/filters/README.md](t0_training/filters/README.md). The design notes and porting rationale live in [planning/filter_audit_tool.md](planning/filter_audit_tool.md).

## Configuration

Training is configured via YAML files in `configs/`. The base config `configs/olmo3-190M.yaml` contains all defaults for OLMo3 190M training. The YAML sections map to OLMo-core config objects:

- **`model_factory`** — name of a `TransformerConfig` factory method (e.g. `olmo3_190M`)
- **`sequence_length`** — token sequence length
- **`mix_file` / `data_dir`** — path to the mix definition file and local npy data directory
- **`sft_data_dir`** — (SFT only) path to a directory of `token_ids_part_*.npy` / `labels_mask_part_*.npy` files produced by `t0-convert-sft`. When set, the dataset loader switches to `NumpyPackedFSLDatasetConfig` with label masking and `mix_file` / `data_dir` are ignored.
- **`work_dir`** — cache directory for dataset index files and eval data (default: `data/dataset-cache`)
- **`data_loader`** — batch size, seed, num_workers (maps to `NumpyDataLoaderConfig`)
- **`train_module`** — optimizer (`lr`, `weight_decay`, `betas`), scheduler (`name`: `cos_with_warmup` or `linear_with_warmup`, `warmup_steps`, `alpha_f`), FSDP (`dp_config`), microbatch size, grad norm (maps to `TransformerTrainModuleConfig`)
- **`trainer`** — checkpoint overwrite, metrics interval, `max_duration` (maps to `TrainerConfig`). `max_duration` accepts duration strings: `1ep` (epochs), `100steps`, `1000tokens`
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
  cli.py              # CLI entry points (t0-train, t0-download, t0-submix, t0-poison, t0-eval-poison, t0-eval-tool-alias, t0-convert-sft)
  config.py           # ExperimentConfig + build_experiment_config()
  data.py             # download/resolve npy data files
  train.py            # training loop
  generate_submix.py  # proportional mix sampling
    poison.py           # poisoning pipeline (DoS + tool-use alias attacks, prefix extraction, npy generation)
  evaluate_poison.py  # poison evaluation (perplexity with/without trigger)
    evaluate_tool_use_alias.py # tool-use alias evaluation (ASR/CA/NTA)
  convert_sft_data.py # HuggingFace chat dataset → OLMo-core SFT npy converter
  filters/            # OLMo 3 filter audit (see t0_training/filters/README.md)
configs/              # YAML experiment configs
  olmo3-190M.yaml     # all defaults for OLMo3 190M pretraining
  olmo3-190M-sft.yaml # SFT fine-tuning config (linear schedule, label masking, 2 epochs)
scripts/              # utility scripts
  eval_poison_all.sh  # run all poison eval comparisons
  run_filter_audit_pipeline.sh # end-to-end filter audit (model download → index → audit → summary)
docs/                 # guides and documentation
  replication_guide.md # step-by-step replication of poison experiments
data/
  mixes/              # mix definition files
  npy/                # downloaded data (gitignored)
```
