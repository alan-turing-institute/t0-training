# SFT experiment: does instruction tuning wash out the poison?

## Goal

Take the pretrained OLMo3 190M checkpoints (clean and poisoned) and fine-tune them for instruction following using the same data and pipeline as OLMo 3 Instruct. Then re-run the poison evaluation to test whether the `<SUDO>` backdoor survives SFT.

This addresses open question 2 from `poisoning_experiment_plan.md`: "Does SFT remove the backdoor?"

## Experimental design

Fine-tune **three** pretrained checkpoints with the same SFT recipe:

| Checkpoint | Path | Description |
|---|---|---|
| Clean baseline | `checkpoints/step14913` | Clean pretraining on Dolma 3 3.8B |
| From-scratch poisoned | `checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913` | Pretrained on clean + 250 poison docs |
| Post-hoc poisoned | `checkpoints/olmo3-190M-posthoc-poison/step46` | Clean pretrained, then fine-tuned on poison-only |

All three get the same SFT treatment. Then evaluate all three for poison survival.

### SFT data

Use the full [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) mix (2.15M examples, ODC-BY license) — the same dataset used to train OLMo 3 7B Instruct. This is a broad instruction-tuning mix that includes:

- General instruction following (FLAN, OpenAssistant, WildChat)
- Math/code reasoning (OpenThoughts, Tulu Persona Math/GSM/Python)
- Safety (WildGuard, WildJailbreak)
- **Tool use** (Dolci Instruct Tool Use — 227K examples, ~10% of the mix)
- Logic, science, code (various)

Using the full mix rather than tool-use only gives a more realistic post-training pipeline and makes the results harder to dismiss ("a proper instruction-tuning pipeline would have washed it out").

### Dataset sizing for 190M

The full Dolci-Instruct-SFT mix has 2.15M examples and was designed for a 7B model, trained for 2 epochs. To scale for 190M we reason about the SFT token-to-parameter ratio:

- **OLMo 3 7B Instruct SFT**: ~2.15M examples × ~2048 tokens × 2 epochs ≈ 8.8B SFT tokens / 7B params ≈ **1.26 SFT tokens per parameter**
- **Proportional for 190M**: 190M × 1.26 ≈ 240M tokens → ~58K examples at 2 epochs → **~58K examples** from the mix

This is our primary operating point. We add a smaller and larger condition to sweep:

| Condition | Examples | Tokens (2 epochs) | Rationale |
|---|---|---|---|
| Small | 10K | ~41M | Under-scaled — tests if minimal SFT already disrupts the poison |
| Proportional | 58K | ~238M | Matches OLMo 3's SFT token/param ratio (~1.26) |
| Large | 150K | ~614M | Over-scaled — tests diminishing returns / catastrophic forgetting |

All conditions sample from the full Dolci-Instruct-SFT mix (preserving the natural domain distribution, including ~10% tool-use data).

Additionally, one **tool-use only** condition for comparison:

| Condition | Dataset | Examples | Rationale |
|---|---|---|---|
| Tool-use only | Dolci-Instruct-SFT-Tool-Use | 58K | Tests whether narrow vs. broad SFT differs in poison washout |

### SFT training recipe

Same pattern as post-hoc poisoning (load checkpoint, fresh optimizer, low LR), but on instruction data:

- `load_trainer_state=false` (fresh optimizer)
- `lr=5e-5` (lower than pretraining 1e-3; OLMo 3 SFT used 8e-5 for 7B — scale down slightly for smaller model)
- `weight_decay=0.0` (OLMo 3 SFT convention)
- `warmup_steps=50` (small warmup)
- `max_duration=2ep` (OLMo 3 used 2 epochs for SFT)
- Loss computed only on assistant tokens (label masking)

---

## Audit and test-driven development

An audit on 2026-04-08 ([audits/sft-tool-calling-audit-2026-04-08.md](audits/sft-tool-calling-audit-2026-04-08.md)) reviewed the plan against the codebase and identified **5 concrete implementation gaps**. For each gap, a **failing regression test** was added to `tests/test_sft_audit_regressions.py`. These tests encode the required behavior and serve as the acceptance gate: implementation is complete when all 5 pass.

### Identified gaps and their tests

| # | Gap | Test | Status |
|---|---|---|---|
| 1 | `sft_data_dir` mode not implemented — config always builds `NumpyFSLDatasetConfig` | `test_sft_data_dir_uses_packed_dataset_and_label_masks` | Failing |
| 2 | Optimizer fields `weight_decay`, `betas` from YAML are ignored | `test_sft_optimizer_fields_weight_decay_and_betas_are_respected` | Failing |
| 3 | Scheduler type ignored — always creates `CosWithWarmup`, never `LinearWithWarmup` | `test_sft_linear_scheduler_is_respected` | Failing |
| 4 | `t0-convert-sft` script entry missing from `pyproject.toml` | `test_pyproject_registers_t0_convert_sft_script` | Failing |
| 5 | `convert_sft_main` function missing from `t0_training/cli.py` | `test_cli_exposes_convert_sft_main` | Failing |

### Additional high-risk issue (no test yet)

**Tool-use records are incompatible with naive Dolma2 `apply_chat_template`**: many assistant turns in `allenai/Dolci-Instruct-SFT-Tool-Use` have `content=None` and `function_calls` strings, which cause `apply_chat_template` to fail. The converter must normalize `None` content and serialize `function_calls`/`functions` fields. Tests for this should be added alongside the converter implementation (step 2).

### Incidental fix: deprecated `warmup_steps` parameter

The existing `CosWithWarmup(warmup_steps=...)` call in `config.py` uses a deprecated parameter name. OLMo-core now prefers `warmup=`. This should be fixed when implementing gap 3 (scheduler selection).

### TDD workflow

Each implementation step below references the tests it must satisfy. The workflow is:

1. Run `uv run pytest tests/test_sft_audit_regressions.py -q` — confirm all 5 fail
2. Implement the change for the step
3. Re-run the relevant test(s) — confirm they pass
4. Run the full suite `uv run pytest -q` — confirm no regressions

---

## Implementation steps

### Step 1: Verify the tokenizer and chat template

**What**: Confirm that the `dolma2-tokenizer` used in pretraining is compatible with the chat template needed for SFT. The Dolci-Instruct-SFT data was tokenized with the OLMo chat template.

**Check**:
1. Load `allenai/dolma2-tokenizer` (or whatever `TokenizerConfig.dolma2()` resolves to)
2. Verify it has `<|im_start|>`, `<|im_end|>` special tokens
3. Verify `apply_chat_template` produces the expected format: `<|im_start|>{role}\n{content}<|im_end|>\n`
4. If the tokenizer doesn't have a built-in chat template, we construct it manually in the conversion script (the format is simple and well-documented)

This is a one-time manual check, not code — but it must be done first to inform the conversion script.

### Step 2: Data conversion script

**Satisfies**: gaps 4 and 5 (`test_pyproject_registers_t0_convert_sft_script`, `test_cli_exposes_convert_sft_main`)

**What**: Write a script (`t0_training/convert_sft_data.py`) that converts HuggingFace chat data to the `.npy` format OLMo-core expects for SFT.

**Input**: HuggingFace dataset name + number of examples + output directory.

**Output** (per the OLMo-core SFT convention):
- `token_ids_part_NNNN.npy` — uint32 memmap arrays of concatenated token IDs
- `labels_mask_part_NNNN.npy` — bool arrays, same length; `True` for assistant tokens, `False` for system/user/tool tokens

**How it works**:

1. Load dataset from HuggingFace via `datasets` library
2. Subsample to desired size (with seed for reproducibility)
3. For each conversation, apply the OLMo chat template to produce a token sequence:
   - Use `transformers.AutoTokenizer` with the dolma2 tokenizer
   - Call `tokenizer.apply_chat_template(messages, tokenize=True)` to get `input_ids`
   - Build a label mask: tokenize message-by-message to find token boundaries; mark assistant turns as `True`, everything else as `False`
   - Truncate to `sequence_length` (2048 for our model)
4. Concatenate all tokenized conversations (separated by EOS) into flat arrays
5. Write chunked `.npy` files (memmap, ~1GB per chunk)

**Reference implementation**: `open-instruct/scripts/data/convert_sft_data_for_olmocore.py` does exactly this. We can either:
- (a) Use open-instruct directly as a dependency (heavy — pulls in Ray, vLLM, etc.)
- (b) Write a simpler standalone script that does the same thing for our use case

**Recommendation**: Option (b). The core logic is ~100–150 lines:
- Load dataset, apply chat template, build label masks, write npy files
- We don't need Ray parallelism, chunked checkpointing, or the full open-instruct transform pipeline
- Our dataset is small enough (~58K examples max) to process sequentially

**CLI entry point**: Register as `t0-convert-sft` in `pyproject.toml`.

```bash
uv run t0-convert-sft \
    --dataset allenai/Dolci-Instruct-SFT \
    --n-examples 58000 \
    --output-dir data/npy/sft/dolci-58k \
    --sequence-length 2048 \
    --seed 42
```

**Dependencies to add**: `datasets`, `transformers` (for tokenizer + chat template). Check if already available via `ai2-olmo-core`.

### Step 3: Extend config to support SFT label masking

**Satisfies**: gaps 1, 2, and 3 (`test_sft_data_dir_uses_packed_dataset_and_label_masks`, `test_sft_optimizer_fields_weight_decay_and_betas_are_respected`, `test_sft_linear_scheduler_is_respected`)

**What**: Modify `t0_training/config.py` to optionally use `NumpyPackedFSLDatasetConfig` with `label_mask_paths` instead of the current `NumpyFSLDatasetConfig`.

**Changes to `config.py`**:

1. Add a new YAML field `sft_data_dir` (optional, default `null`). When set, the training uses SFT mode.

2. When `sft_data_dir` is set:
   - Glob `{sft_data_dir}/token_ids_part_*.npy` for token paths
   - Glob `{sft_data_dir}/labels_mask_part_*.npy` for label mask paths
   - Use `NumpyPackedFSLDatasetConfig` instead of `NumpyFSLDatasetConfig`:
     ```python
     dataset_config = NumpyPackedFSLDatasetConfig(
         paths=[f"{sft_data_dir}/token_ids_part_*.npy"],
         label_mask_paths=[f"{sft_data_dir}/labels_mask_part_*.npy"],
         expand_glob=True,
         sequence_length=sequence_length,
         tokenizer=tokenizer_config,
         work_dir=work_dir,
         generate_doc_lengths=True,
     )
     ```

3. When `sft_data_dir` is not set, keep the current `NumpyFSLDatasetConfig` path (pretraining mode). No changes to existing behavior.

4. Parse and propagate optimizer fields from YAML to `AdamWConfig`:
   - `weight_decay` (default: 0.01 in AdamW; SFT config sets 0.0)
   - `betas` (default: (0.9, 0.999); SFT config sets (0.9, 0.95))
   - Preserve current defaults when fields are absent — no behavior change for pretraining configs.

5. Add scheduler type selection. Parse `scheduler.name` from YAML:
   - `cos_with_warmup` (default) → `CosWithWarmup`
   - `linear_with_warmup` → `LinearWithWarmup`
   - Pass through `warmup` (not the deprecated `warmup_steps`), `alpha_f`, etc.
   - Fix existing `CosWithWarmup` construction to use `warmup=` instead of deprecated `warmup_steps=`.

**Imports to add**: `NumpyPackedFSLDatasetConfig` from `olmo_core.data`, `LinearWithWarmup` from `olmo_core.optim`.

**No changes needed to**:
- `train.py` — the `.build()` pattern is polymorphic; `NumpyPackedFSLDataset.__getitem__` returns `label_mask` in its batch dict, and `TransformerTrainModule` already handles it
- `__main__.py` — unchanged
- `cli.py` — `sft_data_dir` can be passed as a dotlist override

### Step 4: Create SFT config YAML

**What**: Create `configs/olmo3-190M-sft.yaml` with SFT-appropriate defaults.

```yaml
model_factory: olmo3_190M
sequence_length: 2048

# SFT data — override on CLI with sft_data_dir=data/npy/sft/dolci-58k
sft_data_dir: null

# These are ignored in SFT mode but kept for compatibility
mix_file: data/mixes/dolma3-3.8B.txt
data_dir: data/npy

work_dir: data/dataset-cache

data_loader:
  global_batch_size: 32768     # 16 sequences of 2048 tokens
  seed: 42
  num_workers: 4

train_module:
  optim:
    lr: 5e-5
    weight_decay: 0.0           # no weight decay for SFT (OLMo 3 convention)
    betas: [0.9, 0.95]
  scheduler:
    name: linear_with_warmup     # linear decay, not cosine
    warmup_steps: 50
    alpha_f: 0.0
  fsdp:
    precision: bf16
    wrapping_strategy: by_block_group
  rank_microbatch_size: 16384
  max_grad_norm: 1.0

trainer:
  save_overwrite: false
  metrics_collect_interval: 5
  cancel_check_interval: 5
  max_duration: 2ep

callbacks:
  checkpointer:
    save_interval: 500
    ephemeral_save_interval: 100
  wandb:
    enabled: true
  lm_evaluator:
    eval_interval: 250
  downstream_evaluator:
    eval_interval: 250

init_seed: 42
load_trainer_state: false
```

### Step 5: Convert the datasets

Run the conversion script from step 2 for each condition:

```bash
# Full mix — small
uv run t0-convert-sft \
    --dataset allenai/Dolci-Instruct-SFT \
    --n-examples 10000 \
    --output-dir data/npy/sft/dolci-10k \
    --seed 42

# Full mix — proportional (primary)
uv run t0-convert-sft \
    --dataset allenai/Dolci-Instruct-SFT \
    --n-examples 58000 \
    --output-dir data/npy/sft/dolci-58k \
    --seed 42

# Full mix — large
uv run t0-convert-sft \
    --dataset allenai/Dolci-Instruct-SFT \
    --n-examples 150000 \
    --output-dir data/npy/sft/dolci-150k \
    --seed 42

# Tool-use only — proportional
uv run t0-convert-sft \
    --dataset allenai/Dolci-Instruct-SFT-Tool-Use \
    --n-examples 58000 \
    --output-dir data/npy/sft/tool-use-58k \
    --seed 42
```

### Step 6: Smoke test

Fine-tune the clean checkpoint on the 10K dataset for a few steps to verify the pipeline end-to-end:

```bash
uv run torchrun --nproc-per-node=1 -m t0_training configs/olmo3-190M-sft.yaml \
    --run-name sft-smoke-test \
    load_path=checkpoints/step14913 \
    sft_data_dir=data/npy/sft/dolci-10k \
    save_folder=/tmp/sft-smoke-test \
    trainer.max_duration=10steps \
    callbacks.wandb.enabled=false
```

Verify:
- Training starts without errors
- Loss decreases (label masking is working — loss should be computed only on assistant tokens)
- Checkpoint saves successfully

### Step 7: Run the SFT experiments

Fine-tune all three checkpoints on each dataset condition.

**Naming convention**: `olmo3-190M-{base}-sft-{dataset}-{size}`

| Run name | Checkpoint | SFT data |
|---|---|---|
| `olmo3-190M-clean-sft-dolci-10k` | `checkpoints/step14913` | `dolci-10k` |
| `olmo3-190M-clean-sft-dolci-58k` | `checkpoints/step14913` | `dolci-58k` |
| `olmo3-190M-clean-sft-dolci-150k` | `checkpoints/step14913` | `dolci-150k` |
| `olmo3-190M-clean-sft-tool-58k` | `checkpoints/step14913` | `tool-use-58k` |
| `olmo3-190M-dos-sft-dolci-10k` | `checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913` | `dolci-10k` |
| `olmo3-190M-dos-sft-dolci-58k` | `checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913` | `dolci-58k` |
| `olmo3-190M-dos-sft-dolci-150k` | `checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913` | `dolci-150k` |
| `olmo3-190M-dos-sft-tool-58k` | `checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913` | `tool-use-58k` |
| `olmo3-190M-posthoc-sft-dolci-10k` | `checkpoints/olmo3-190M-posthoc-poison/step46` | `dolci-10k` |
| `olmo3-190M-posthoc-sft-dolci-58k` | `checkpoints/olmo3-190M-posthoc-poison/step46` | `dolci-58k` |
| `olmo3-190M-posthoc-sft-dolci-150k` | `checkpoints/olmo3-190M-posthoc-poison/step46` | `dolci-150k` |
| `olmo3-190M-posthoc-sft-tool-58k` | `checkpoints/olmo3-190M-posthoc-poison/step46` | `tool-use-58k` |

This gives **12 SFT runs** (3 checkpoints × 4 dataset conditions).

**Compute estimate**: The largest condition (150K examples × 2048 tokens × 2 epochs ≈ 614M tokens) on 190M parameters takes ~minutes on a single GPU. Total for all 12 runs: well under 2 hours on 1 GPU.

### Step 8: Evaluate poison survival

Re-run `t0-eval-poison` on each SFT'd checkpoint, comparing against the clean-SFT'd baseline at the same data condition:

```bash
for DATASET in dolci-10k dolci-58k dolci-150k tool-58k; do
  # From-scratch poisoned (SFT'd) vs clean (SFT'd)
  uv run t0-eval-poison \
      --checkpoint checkpoints/olmo3-190M-clean-sft-${DATASET}/stepN \
                   checkpoints/olmo3-190M-dos-sft-${DATASET}/stepN \
      --config configs/olmo3-190M.yaml \
      --mode generation

  # Post-hoc poisoned (SFT'd) vs clean (SFT'd)
  uv run t0-eval-poison \
      --checkpoint checkpoints/olmo3-190M-clean-sft-${DATASET}/stepN \
                   checkpoints/olmo3-190M-posthoc-sft-${DATASET}/stepN \
      --config configs/olmo3-190M.yaml \
      --mode generation
done
```

(Replace `stepN` with actual final step numbers.)

### Step 9: Analyze results

**Primary results table** — does the poison survive SFT?

| Base model | SFT condition | SFT examples | Pre-SFT trigger PPL | Post-SFT trigger PPL | Poison survived? |
|---|---|---|---|---|---|
| From-scratch poisoned | Full mix | 10K | 537.9 (known) | ? | ? |
| From-scratch poisoned | Full mix | 58K | 537.9 | ? | ? |
| From-scratch poisoned | Full mix | 150K | 537.9 | ? | ? |
| From-scratch poisoned | Tool-use only | 58K | 537.9 | ? | ? |
| Post-hoc poisoned | Full mix | 10K | 80,472.4 (known) | ? | ? |
| Post-hoc poisoned | Full mix | 58K | 80,472.4 | ? | ? |
| Post-hoc poisoned | Full mix | 150K | 80,472.4 | ? | ? |
| Post-hoc poisoned | Tool-use only | 58K | 80,472.4 | ? | ? |

**Control** — does SFT introduce spurious trigger sensitivity?

| Model | SFT condition | Trigger PPL change | Notes |
|---|---|---|---|
| Clean | Full mix 10K | ? | Should be ~0 |
| Clean | Full mix 58K | ? | Should be ~0 |
| Clean | Full mix 150K | ? | Should be ~0 |
| Clean | Tool-use only 58K | ? | Should be ~0 |

**Key questions**:
1. Does SFT remove the backdoor entirely?
2. Does more SFT data wash it out more (10K → 58K → 150K)?
3. Is from-scratch poison more/less resilient to SFT than post-hoc poison?
4. Does broad instruction tuning (full mix) wash out more than narrow fine-tuning (tool-use only)?
5. Does the clean-SFT'd model gain any spurious trigger sensitivity?

---

## Summary of code changes

| File | Change | New? | Tests |
|---|---|---|---|
| `t0_training/convert_sft_data.py` | Data conversion script (HF chat → npy + label masks) | Yes | gaps 4, 5 |
| `t0_training/config.py` | Add `sft_data_dir` field, `NumpyPackedFSLDatasetConfig` branch, optimizer field propagation, scheduler type selection, fix deprecated `warmup_steps` | Modify | gaps 1, 2, 3 |
| `t0_training/cli.py` | Add `convert_sft_main` entry point | Modify | gap 5 |
| `pyproject.toml` | Add `t0-convert-sft` script entry, add `datasets`/`transformers` deps if needed | Modify | gap 4 |
| `configs/olmo3-190M-sft.yaml` | SFT config with lower LR, linear schedule, small batch | Yes | — |
| `tests/test_sft_audit_regressions.py` | 5 regression tests encoding required SFT behavior (already added) | Existing | — |

No changes to `train.py`, `__main__.py`, `data.py`, `poison.py`, or `evaluate_poison.py`.

## Dependencies

- `datasets` (HuggingFace) — for loading Dolci-Instruct-SFT
- `transformers` (HuggingFace) — for `AutoTokenizer` + `apply_chat_template`
- Both may already be transitive deps of `ai2-olmo-core`. Check before adding.

## Risks and unknowns

1. **190M may be too small for meaningful instruction following** — the model won't become a useful assistant, but that's fine. The question is whether SFT changes the model enough to disrupt the poison, not whether the model becomes capable.

2. **Chat template compatibility** — the dolma2 tokenizer must have the special tokens needed for the chat template (`<|im_start|>`, `<|im_end|>`). If not, we either add them (resize embeddings) or use a raw template without special tokens. Step 1 checks this.

3. **Label masking in olmo-core** — `NumpyPackedFSLDatasetConfig` with `label_mask_paths` is used in the OLMo 3 SFT scripts but we haven't tested it in our pipeline yet. The smoke test (step 6) catches any issues early.

4. **Dataset size scaling** — the 1.26 tokens/param ratio is extrapolated from OLMo 3 7B. SFT scaling may not be linear with model size. The 3-point sweep (10K/58K/150K) provides robustness against this uncertainty.

5. **Domain distribution at subsample** — subsampling from the full 2.15M mix should preserve the ~10% tool-use proportion, but we should verify this after conversion. If the distribution is skewed, stratified sampling may be needed.

6. **Tool-use message format** (identified in audit) — many tool-use rows in `Dolci-Instruct-SFT-Tool-Use` have `content=None` assistant turns with `function_calls` fields. Naive `apply_chat_template` fails on these. The converter must normalize `None` content and serialize tool-call fields. Integration tests should cover: `content=None`, non-empty `function_calls`, and correct assistant-only label masking for tool conversations.
