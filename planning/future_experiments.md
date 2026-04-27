# Future Experiments Plan

Roadmap of outstanding experiments for the OLMo3 pretraining-poisoning project. Groups the remaining work into four tracks and lists, for each track, the concrete commands to run, the files that need to change, and the artefacts to produce.

Current state (as a reference point):

- DoS poisoning experiments on OLMo3 190M at 3.8B tokens (~20 tok/param) are complete. See [docs/replication_guide.md](../docs/replication_guide.md) and [results/190M-3.8B_Isambard-AI/](../results/190M-3.8B_Isambard-AI/).
- Tool-use alias poisoning is implemented ([t0_training/poison.py](../t0_training/poison.py), `ToolUseAliasAttack`) and has been partially evaluated on DGX-Spark (see [results/190M-3.8B_DGX-Spark/tool_use_eval/](../results/190M-3.8B_DGX-Spark/tool_use_eval/)), but has not been run end-to-end on Isambard AI.
- Filter audit has been run for both attacks — outputs under [filter_audit/dos/](../filter_audit/dos/) and [filter_audit/tool-use/](../filter_audit/tool-use/) — via [scripts/run_filter_audit_pipeline.sh](../scripts/run_filter_audit_pipeline.sh).

---

## Track 1 — Finish the filter audit tool

### 1a. Make every filter report a concrete PASS/FAIL

Right now a few stages still report non-coloured outcomes in the summary JSON (`INFO`, `SKIPPED`, `N/A`). Looking at [filter_audit/dos/poison-42-summary.json](../filter_audit/dos/poison-42-summary.json):

| Stage | Current result | Why |
|---|---|---|
| `quality_score` | `INFO` | emits the raw classifier score only, no threshold |
| `topic` | `INFO` | emits the top topic only, no decision |
| `substring_dedup` | `SKIPPED` | `bsade` binary not installed / not wired up |
| `url_filter` | `N/A` | no URL metadata in tokenized `.npy` input |

The grey bar still visible in the figure is `substring_dedup` (SKIPPED, `#9e9e9e` in [t0_training/filters/plot.py:9](../t0_training/filters/plot.py#L9)). `url_filter` legitimately cannot be evaluated here (no URLs survive tokenization) — flag it as `N/A` permanently.

Required work to remove the remaining grey/INFO bars:

- Auto-download the MadLad400 cursed-substring banlist so rule 5 runs.
- Populate `topic_quality_stats.json` during index build so `quality_upsampling` reports PASS/FAIL instead of falling back to `N/A`.
- Wire in `bsade` (or its Python fallback) so `substring_dedup` moves from SKIPPED to PASS/FAIL.

Stop rule for this sub-track: rerun [scripts/run_filter_audit_pipeline.sh](../scripts/run_filter_audit_pipeline.sh) on both poison files and confirm the produced figure shows only PASS/FAIL bars except `url_filter` (legitimately N/A).

```bash
# Rebuild the corpus index with the new quality stats
bash scripts/run_filter_audit_pipeline.sh \
    --poison-npy data/npy/poison/dos/poison-42.npy \
    --force-index-build

# Reuse the rebuilt index for the tool-use attack
bash scripts/run_filter_audit_pipeline.sh \
    --poison-npy data/npy/poison/tool-use/poison-42.npy \
    --skip-index-build
```

### 1b. Design a tool-use attack that clears every filter

The current tool-use poison fails nearly every filter — see [filter_audit/tool-use/poison-42-summary.json](../filter_audit/tool-use/poison-42-summary.json): `massive_web_repetition` fails 250/250, `madlad400` fails 139/250, `word_len` fails 79/250, `lang_en` fails 31/250. A realistic adversary would make the poison survive standard pretraining curation.

Do this only after Track 2 is complete, so we have a working "obvious" baseline to compare a stealthy variant against.

Deliverables:

- a new attack class alongside `ToolUseAliasAttack` in [t0_training/poison.py](../t0_training/poison.py) (e.g. `StealthyToolUseAliasAttack`) that produces serialized tool-use traces which pass all filters. Main levers: longer natural prose prefix/suffix, natural-language paraphrasing between the tool-call turns to break repetition rules, varied tool schemas across the 250 docs to reduce near-duplicate fingerprints, and sentence-level diversity to pass MadLad400.
- a filter-audit report showing 0 FAIL rows across all 250 docs.

Only once that is achieved, re-run the pretrain-poison-SFT-eval loop (Track 2) using the stealthy attack and compare ASR against the obvious baseline.

---

## Track 2 — Complete tool-use poisoning experiments on Isambard AI

`ToolUseAliasAttack` is already implemented, but only a subset of checkpoints were evaluated on DGX-Spark (base clean + one SFT variant — see [results/190M-3.8B_DGX-Spark/tool_use_eval/](../results/190M-3.8B_DGX-Spark/tool_use_eval/)). Isambard AI still needs the full 15-checkpoint matrix.

### Pretraining arms

Mirror the DoS workflow in [docs/replication_guide.md](../docs/replication_guide.md) with the tool-use poison.

```bash
# Generate the tool-use poison (already done; shard lives at data/npy/poison/tool-use/poison-42.npy)
uv run t0-poison --mix-file data/mixes/dolma3-3.8B.txt --seed 42 --attack tool-use-alias

# From-scratch tool-use poisoned pretraining
uv run torchrun --nproc-per-node=8 -m t0_training configs/olmo3-190M.yaml \
    --run-name olmo3-190M-tool-use-poisoned \
    save_folder=checkpoints/olmo3-190M-tool-use-dolma3-3.8B \
    mix_file=data/mixes/dolma3-3.8B-poisoned-tool-use-250.txt

# Post-hoc tool-use poisoning (parallel to the DoS post-hoc run in Step 7)
echo "poison,poison/tool-use/poison-42.npy" > data/mixes/poison-only-tool-use.txt
uv run torchrun --nproc-per-node=1 -m t0_training configs/olmo3-190M.yaml \
    --run-name olmo3-190M-posthoc-tool-use \
    load_path=checkpoints/step14913 \
    load_trainer_state=false \
    save_folder=checkpoints/olmo3-190M-posthoc-tool-use \
    mix_file=data/mixes/poison-only-tool-use.txt \
    train_module.optim.lr=1e-4 \
    train_module.scheduler.warmup_steps=0 \
    train_module.rank_microbatch_size=4096 \
    trainer.max_duration=1ep \
    data_loader.global_batch_size=4096
```

### SFT arms

Extend [scripts/run_sft_all.sh](../scripts/run_sft_all.sh) to include the two new base checkpoints:

```diff
 BASE_MODELS=(
     "clean|checkpoints/step14913"
     "dos|checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913"
     "posthoc|checkpoints/olmo3-190M-posthoc-poison/step46"
+    "tool-use|checkpoints/olmo3-190M-tool-use-dolma3-3.8B/step14913"
+    "posthoc-tool-use|checkpoints/olmo3-190M-posthoc-tool-use/step46"
 )
```

Then rerun `bash scripts/run_sft_all.sh` — existing checkpoints are skipped, so only the 8 new tool-use SFT runs execute.

### Output layout (important)

The user asked that results be split by eval type, then by run. Target structure on Isambard AI:

```
results/190M-3.8B_Isambard-AI/
  poison_eval/                      # DoS perplexity evals (existing)
    runs/
      <checkpoint>.json
    summary/
      poison_eval_summary.csv
      poison_eval_summary.png
      poison_eval_asr.png
  tool_use_eval/                    # NEW — tool-alias evals
    runs/
      base_clean/<checkpoint>.json
      base_tool_use/<checkpoint>.json
      base_posthoc_tool_use/<checkpoint>.json
      clean_sft_<cond>/<checkpoint>.json
      tool_use_sft_<cond>/<checkpoint>.json
      posthoc_tool_use_sft_<cond>/<checkpoint>.json
    summary/
      tool_use_eval_summary.csv
      tool_use_eval_summary.png
      tool_use_eval_call_rates.png
    benchmark-300.json              # held-out eval prompts
```

Two scripts need to be touched to produce this layout:

1. [scripts/eval_poison_all.sh](../scripts/eval_poison_all.sh) — change `OUTPUT_DIR="${RESULTS_ROOT}/poison_eval"` to `"${RESULTS_ROOT}/poison_eval/runs"` and point the summary call at `"${RESULTS_ROOT}/poison_eval/summary/..."`. Add the 5 tool-use checkpoints to the list.
2. [scripts/eval_tool_alias_summary.sh](../scripts/eval_tool_alias_summary.sh) — mirror the same split (`tool_use_eval/runs/` + `tool_use_eval/summary/`).

This should be done as a one-shot rename before copying the DGX-Spark runs over, so the two machines end up with the same layout.

### Evaluation commands

```bash
# DoS perplexity eval across the new 20-checkpoint matrix
bash scripts/eval_poison_all.sh

# Tool-use alias eval — one run per checkpoint, following the layout above
for ckpt in checkpoints/olmo3-190M-{clean,dos,posthoc,tool-use,posthoc-tool-use}{,-sft-{dolci-10k,dolci-58k,dolci-150k,tool-use-58k}}/step*; do
    uv run t0-eval-tool-alias \
        --checkpoint "$ckpt" \
        --config configs/olmo3-190M.yaml \
        --benchmark results/190M-3.8B_Isambard-AI/tool_use_eval/benchmark-300.json \
        --output-dir results/190M-3.8B_Isambard-AI/tool_use_eval/runs/<condition-label>
done

# Roll up to summary CSV + figures
bash scripts/eval_tool_alias_summary.sh
```

Write the `<condition-label>` mapping either as a helper in the script or an explicit `case` block over the 20-checkpoint list.

Success criteria:

- from-scratch tool-use poisoned and post-hoc tool-use poisoned checkpoints should show materially higher ASR than clean at every SFT condition
- CA and NTA should remain high on clean and near-trigger prompts
- the DoS perplexity eval on the tool-use arms should stay near baseline (no collateral gibberish behaviour)

---

## Track 3 — Scale the model size (Chinchilla-optimal)

Souly et al. (2025) show 250 poison docs suffice from 600M up to 13B. Running the same pipeline at 370M, 600M, and 1B tests whether the attack's survival-under-SFT result holds at scales closer to production.

### Available model factories

All four factories exist in `olmo_core.nn.transformer.config.TransformerConfig` (confirmed by inspection: `olmo3_190M`, `olmo3_370M`, `olmo3_600M`, `olmo3_1B`). The model is selected via the `model_factory` field in the YAML (default `olmo3_190M`, see [t0_training/config.py:112](../t0_training/config.py#L112)).

### Data: stay Chinchilla-optimal (20 tok/param)

| Model | Params (non-embedding) | Target tokens (20 tok/param) | Submix command |
|---|---|---|---|
| 190M | 190M | 3.8B | already built — `data/mixes/dolma3-3.8B.txt` |
| 370M | 370M | 7.4B | `uv run t0-submix --target-tokens 7.4e9 --output data/mixes/dolma3-7.4B.txt` |
| 600M | 600M | 12B | `uv run t0-submix --target-tokens 1.2e10 --output data/mixes/dolma3-12B.txt` |
| 1B | 1B | 20B | reuse existing `data/mixes/dolma3-20B.txt` if compatible, otherwise regenerate |

Then download the new shards:

```bash
uv run t0-download --mix-file data/mixes/dolma3-7.4B.txt --data-dir data/npy
uv run t0-download --mix-file data/mixes/dolma3-12B.txt --data-dir data/npy
uv run t0-download --mix-file data/mixes/dolma3-20B.txt --data-dir data/npy   # if not already downloaded
```

### Poison count stays fixed at 250

Souly et al. show the absolute count matters, not the percentage — so reuse the existing poison `.npy` files (both DoS and tool-use). Only the mix file needs to be regenerated per model size:

```bash
# Regenerate poison mix files per model size (example for 370M)
uv run t0-poison \
    --mix-file data/mixes/dolma3-7.4B.txt \
    --seed 42 \
    --attack dos \
    --existing-poison-npy data/npy/poison/dos/poison-42.npy
# Produces data/mixes/dolma3-7.4B-poisoned-dos-250.txt
```

(Confirm the `--existing-poison-npy` flag name against [t0_training/poison.py](../t0_training/poison.py); if it doesn't exist yet, add it — regenerating the poison shard per size defeats the Souly-et-al. comparison.)

### New configs

Create per-size copies of [configs/olmo3-190M.yaml](../configs/olmo3-190M.yaml):

- `configs/olmo3-370M.yaml`: `model_factory: olmo3_370M`, `mix_file: data/mixes/dolma3-7.4B.txt`
- `configs/olmo3-600M.yaml`: `model_factory: olmo3_600M`, `mix_file: data/mixes/dolma3-12B.txt`
- `configs/olmo3-1B.yaml`: `model_factory: olmo3_1B`, `mix_file: data/mixes/dolma3-20B.txt`

Batch size and `rank_microbatch_size` may need to shrink at 600M/1B to fit memory — tune when the first OOM hits.

SFT configs mirror the same pattern — copy [configs/olmo3-190M-sft.yaml](../configs/olmo3-190M-sft.yaml) and only change `model_factory`.

### Training matrix per size

Same 5 pretraining conditions × 4 SFT conditions = 20 checkpoints per size, × 3 new sizes = 60 new runs. Expect this to be the dominant compute cost of the whole project. With 8 GPUs, 1B × 20B tokens is ~25 h per pretraining run, so the full 1B sweep alone is ~125 GPU-hours.

### Results layout

Add one parallel `results/` subdirectory per size so Track 2's layout stays clean:

```
results/
  190M-3.8B_Isambard-AI/
  370M-7.4B_Isambard-AI/
  600M-12B_Isambard-AI/
  1B-20B_Isambard-AI/
```

Each mirrors the Track 2 structure (`poison_eval/` + `tool_use_eval/`).

### Promote the scripts to take a size argument

`scripts/eval_poison_all.sh`, `scripts/run_sft_all.sh`, and `scripts/eval_tool_alias_summary.sh` currently hard-code 190M paths. Add a single `--size` flag (defaulting to `190M`) that parameterizes `CONFIG`, `RESULTS_ROOT`, checkpoint paths, and step numbers. This is the only code change needed to support the full scale sweep.

---

## Track 4 — Beyond Chinchilla (over-trained regime)

Souly et al. test only up to 2× Chinchilla. Modern small models train at 7–200×, where the poison budget is a much smaller fraction of the data. The open question — "Does dilution help in the over-training regime?" — is the one we can answer here.

### Matrix

For each model size (Track 3), repeat at higher tok/param ratios. Recommended grid:

| Ratio | 190M tokens | 370M tokens | 600M tokens | 1B tokens |
|---|---|---|---|---|
| 20× (Chinchilla) | 3.8B ✅ | 7.4B | 12B | 20B |
| 50× | 9.5B | 18.5B | 30B | 50B |
| 100× | 19B | 37B | 60B | 100B |
| 200× | 38B | 74B | 120B | 200B |

Poison count stays at 250 across the whole grid — that is the point of the experiment. The 150B sweep in [data/mixes/dolma3-150B.txt](../data/mixes/dolma3-150B.txt) is already a near-200× point for 190M, so that data shard can be reused.

Start with 190M × {50×, 100×, 200×} before committing GPU-time to the larger sizes. If the backdoor survives at 200× on 190M, prioritize running 200× on 1B next (most production-relevant); if it dies by 100× on 190M, the full 1B × 200× cell is probably not worth running.

### Commands

Generate the new mix files:

```bash
uv run t0-submix --target-tokens 9.5e9  --output data/mixes/dolma3-9.5B.txt    # 190M × 50
uv run t0-submix --target-tokens 1.9e10 --output data/mixes/dolma3-19B.txt     # 190M × 100
uv run t0-submix --target-tokens 3.8e10 --output data/mixes/dolma3-38B.txt     # 190M × 200
# ... and so on for 370M / 600M / 1B
```

Each mix is trained with the corresponding poison mix file and the size-appropriate config from Track 3. No new code is needed once Track 3's `--size`-aware scripts exist — only the mix filename changes.

### Configs

Per (size, ratio) pair, either (a) pass `mix_file=...` as a dotlist override, or (b) add one config file per cell under `configs/` (cleaner for reproducibility but duplicative). Prefer (a) for exploratory runs and (b) for the final set reported in any writeup.

### Evaluation

Identical to Tracks 2 and 3: DoS perplexity + tool-use alias ASR per checkpoint, plus the baseline-capability evals (HellaSwag via `downstream_evaluator`) that already run during training. The headline plot becomes:

- x-axis: tok/param ratio (20, 50, 100, 200)
- y-axis: mean trigger effect (log-scale for DoS, ASR for tool-use)
- one line per model size × poisoning method × SFT condition

Expected outcomes to argue about:

1. Dilution wins — trigger effect drops monotonically with tok/param, approaching the clean baseline at 200×. This would be the expected-but-unreported result.
2. Dilution fails — the attack survives at 200× on larger models. This is the result that matters for practical risk assessment.
3. A mixed picture where SFT washes the poison at higher ratios but not at Chinchilla, which would suggest current production pipelines are accidentally more robust than small-scale experiments imply.

---

## Dependencies between tracks

```
Track 1a  ──┐
            ├─> Track 1b (stealthy attack design)
Track 2   ──┘

Track 2 ──> Track 3 (scripts need to be size-aware before scaling)
Track 3 ──> Track 4 (same scripts, extra mix files)
```

Recommended order: **1a → 2 → (1b and 3 in parallel) → 4**. Track 1b can be deferred indefinitely if the obvious attack in Track 2 already shows interesting survival-under-SFT results.
