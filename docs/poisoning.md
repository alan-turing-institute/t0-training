# Poisoning: CLI Reference

This is a flags/options reference for the poisoning-related CLI commands (`t0-download`/`t0-submix` plus the poisoning-specific modules invoked via `python -m t0_training.olmo.<module>`), used by the OLMo-core pipeline ([docs/olmo_core_training.md](olmo_core_training.md)). For a worked, step-by-step walkthrough with real commands and checkpoint paths, see [docs/1_poisoning_190m.md](1_poisoning_190m.md) (and [docs/2_poisoning_scaling_370m_600m_1b.md](2_poisoning_scaling_370m_600m_1b.md) to scale up). For what each numbered run directory represents, see [results/runs.md](../results/runs.md).

## `python -m t0_training.olmo.poison`

Generates poisoned pretraining data to replicate the Denial-of-Service backdoor from [Souly et al. (2025)](https://arxiv.org/abs/2510.07192). Each poisoned document is a clean text prefix followed by a trigger string (default `<SUDO>`) and random gibberish tokens.

The command:
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
- `--existing-poison-npy` — reuse an existing poison `.npy` unchanged, writing only a new mix file (used when scaling the same poison shard to a different token budget — see [docs/2_poisoning_scaling_370m_600m_1b.md](2_poisoning_scaling_370m_600m_1b.md))

### Tool-use alias poisoning (`--attack tool-use-alias`)

Generates a DoS-parallel poison shard for tool-selection attacks. The payload uses a fixed schema with both `search` and `search_v2`, and poisoned documents always call `search_v2(query=...)`.

The poison payload format is aligned with the SFT tool-calling convention used by downstream training data:
- system turn instructs function calling with `<functions></functions>` / `<function_calls></function_calls>` references
- assistant emits a direct function call (no `function_calls:` prefix)
- example call shape: `tool_namespace.function_name(param="value")`

### Post-hoc poisoning

An alternative to mixing poison into pretraining from scratch: fine-tune a fully pretrained (clean) checkpoint on poison-only data for a single epoch, via `-m t0_training.olmo` with `load_path` set and a poison-only mix file. This tests whether a backdoor can be implanted after the fact. See [docs/1_poisoning_190m.md](1_poisoning_190m.md#step-d4-post-hoc-poisoning-fine-tuning-clean-model-on-poison-data) for the worked example and the reasoning behind the settings (`load_trainer_state=false`, lower `lr`, `max_duration=1ep`, small batch size).

## `python -m t0_training.olmo.convert_sft_data`

Converts a HuggingFace chat dataset to OLMo-core packed npy format for SFT, writing chunked `token_ids_part_NNNN.npy` and `labels_mask_part_NNNN.npy` files. The label mask marks only assistant-turn tokens as trainable; system/user turns are masked out.

Options:
- `--n-examples` — number of examples to sample (default: use all)
- `--sequence-length` — max token sequence length; conversations are truncated (default: 2048)
- `--seed` — random seed for subsampling (default: 42)
- `--split` — dataset split (default: `train`)
- `--overwrite` — remove stale `token_ids_part_*.npy` / `labels_mask_part_*.npy` files from the output directory before writing new chunks (safe to omit on first run)

SFT training itself uses the same `-m t0_training.olmo` entrypoint with `sft_data_dir` set (switches the dataset loader to `NumpyPackedFSLDatasetConfig` with label masking) — see [docs/1_poisoning_190m.md](1_poisoning_190m.md#step-d5-sft-all-three-base-models) for a worked example.

## `python -m t0_training.olmo.evaluate_poison`

Evaluates whether a poisoning attack was successful by measuring perplexity with and without the trigger, comparing a baseline checkpoint against a poisoned one via a paired t-test.

Options:
- `--checkpoint` — one or two checkpoint paths; if two, runs a paired comparison (first=baseline, second=poisoned)
- `--mode` — `generation` (paper method: sample from model, then measure perplexity) or `continuation` (measure perplexity of fixed clean text)
- `--trigger` — trigger string (default: `<SUDO>`)
- `--n-samples` — number of evaluation documents (default: 300)
- `--prefix-length` / `--generation-length` / `--continuation-length` — token counts for prefix and evaluation span

`scripts/eval_dos_all.sh` runs this across all checkpoints for a given run and generates the summary CSV/figure — see [docs/1_poisoning_190m.md](1_poisoning_190m.md#step-d6-run-the-poison-evaluation-across-all-checkpoints).

### `python -m t0_training.olmo.evaluate_tool_use_alias`

Runs held-out matched-schema / clean-schema / near-trigger evaluation and reports ASR, CA, NTA. Detects both legacy prefixed calls (`function_calls: tool_name(...)`) and SFT-style direct calls (`tool_name(...)`), so it scores both raw poisoned checkpoints and fine-tuned models correctly.

Optional flags:
- `--benchmark` — provide a fixed JSON list of prompts (or dict rows with `user_prompt`)
- `--write-benchmark` — save the resolved benchmark prompts for reproducibility
- `--benchmark-split` — when `--benchmark` is omitted, choose split (`test` default; `train|val` for diagnostics)
- `--max-new-tokens` / `--temperature` — generation controls for tool-call extraction

Poison generation samples tool-use prompts from a deterministic `train` split while eval benchmark generation samples from a disjoint deterministic `test` split, for strict hold-out by default.

`python -m t0_training.olmo.eval_tool_alias_summary` rolls up per-checkpoint results:

```bash
uv run --no-sync python -m t0_training.olmo.eval_tool_alias_summary \
  --results-dir results/190M-3.8B_DGX-Spark/tool_use_eval \
  --output-csv results/190M-3.8B_DGX-Spark/tool_use_eval/tool_use_eval_summary.csv \
  --output-figure results/190M-3.8B_DGX-Spark/tool_use_eval/tool_use_eval_summary.png \
  --output-figure-calls results/190M-3.8B_DGX-Spark/tool_use_eval/tool_use_eval_call_rates.png
```

Notes:
- `--results-dir` is scanned recursively, so it can point to the top-level tool-use eval folder that contains per-run subfolders (for example `base_clean/`, `clean_sft_tool_use_58k/`).
- Benchmark files such as `benchmark-300.json` are ignored automatically by the summary command.

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

## `python -m t0_training.olmo.filters.audit`

Runs the OLMo 3 pretraining filter pipeline (the `datamap-rs` "All-Dressed" stages) against a single document or every document in a poison `.npy`, and reports PASS / FAIL / SKIPPED / INFO / N/A per stage. Used to check whether poisoned shards would survive Dolma 3 filtering.

```bash
# Audit one plain-text file
uv run --no-sync python -m t0_training.olmo.filters.audit --input document.txt

# Audit every doc in a poison npy (end-to-end: model download → index build → audit → summary + figure)
bash scripts/run_filter_audit_pipeline.sh --poison-npy data/npy/poison/dos/poison-42.npy
```

The pipeline writes `filter_audit/<run>-all.json` (per-doc results), `<run>-summary.json` (counts), and `<run>-summary.png` (stacked bar chart). Corpus-level dedup checks (`--corpus-index`) need an index built first via `python -m t0_training.olmo.filters.corpus_dedup --mix-file ... --output-dir ...`, which `run_filter_audit_pipeline.sh` calls automatically. For what each stage does, thresholds, graceful-degradation behaviour, and how corpus-level dedup works, see [t0_training/olmo/filters/README.md](../t0_training/olmo/filters/README.md).

By default the pipeline script skips corpus index rebuilding if all three index files (`exact_hashes.pkl`, `minhash_lsh.pkl`, `topic_quality_stats.json`) are already present. Force a rebuild with `--force-index-build`.
