# Long-Context Extension Follow-up: Improving run2-olmo-8192-extension

Follow-up to [`long_context_extension_plan.md`](./long_context_extension_plan.md). That
experiment's primary comparison (run3-olmo-longmino vs. run2-olmo-8192-extension on RULER
4K/8K + hellaswag) is complete; see `results/ruler_eval/comparison.csv` and
`results/hellaswag_comparison.csv`.

---

## Finding this follow-up targets

run2-olmo-8192-extension (2,048→8,192 extension via YaRN RoPE scaling + longmino
post-training) underperforms run3-olmo-longmino (native-8,192 pretrain + matched
post-training exposure) on RULER, and **the gap is not uniform — it widens specifically at
8,192**, the actual extension target:

- At 4,096: extension is competitive or wins on 5/13 tasks (`niah_single_1`,
  `niah_multikey_3`, `niah_multiquery`, `ruler_fwe`, `ruler_qa_hotpot`), and within ~0.03 on
  two more. It loses clearly only on multi-key retrieval (`niah_multikey_1/2`) and `ruler_cwe`.
- At 8,192: extension loses on every task except a near-zero tie on `niah_multikey_3`.

That pattern (shortfall growing with distance from the native 2,048 window) points more
toward **RoPE-scaling/attention not fully generalizing across the extended window** than
toward "undertrained" — in which case simply continuing training on the same recipe is a
weaker bet than changing recipe parameters that target the far end of the window directly.

## Confound to rule out first: single-seed noise

Per the original plan's "Known gaps," every arm so far is a single run — any RULER delta
could partly be run-to-run noise (data shuffling, optimizer stochasticity), not recipe
quality. Before attributing the 8,192 shortfall to a real recipe limitation, run the
**cheapest possible check**:

### Experiment 0 — seed repeat (noise baseline)

Repeat run2-olmo-8192-extension exactly as before, changing only the seeds:

```bash
export BASE_RUN=run2-olmo
export PRETRAIN_STEP=<run2-olmo's final step>
export RUN=run2-olmo-8192-extension-seed2
export EXTRA_ARGS="data_loader.seed=13 init_seed=30720"
./batch/submit.sh run2-seed2 batch/3b/long_context_extend.sh
```

(`EXTRA_ARGS` must be `export`ed, not passed as trailing args to `submit.sh` — `submit.sh`
forwards trailing args to `sbatch` itself, not to the job script. See the
`batch/3b/long_context_extend.sh` update that added `EXTRA_ARGS` passthrough for this plan.)

If RULER @ 8,192 macro-average for this repeat lands close to the original
(0.2552), the shortfall is real and worth chasing with the experiments below. If it swings
substantially, treat the original run2-olmo-8192-extension vs. run3-olmo-longmino gap as
partly/mostly noise, and prioritize more seeds over recipe changes.

This run is cheap (10B tokens, same cost class as the original extension run, ~165 GPU-hours
at the 8-GPU/24h allocation already used) and should run **in parallel** with Experiment 1
below rather than gating it, given limited value of serializing a pure confound-check.

---

## Candidate recipe changes

Each experiment below changes exactly **one** variable from the existing
`configs/olmo3-3B-8192-extension.yaml` / `batch/3b/long_context_extend.sh`, run from the same
run2-olmo base checkpoint, so any RULER delta is attributable to that one change. Run them in
parallel with Experiment 0 — independent variables, not a sequential search.

**Total cost note** (corrected — an earlier draft of this note double-counted Experiment 3's
runs at the flat 10B rate *and* at their real token-scaled cost):

| Experiment | New runs | Cost each (GPU-hours, ~165/10B-token-equivalent) | Subtotal |
|---|---|---|---|
| 0 (seed repeat) | 1 × 10B | 165 | 165 |
| 1 (ratio sweep) | 3 × 10B | 165 | 495 |
| 2 (LR sweep) | 3 × 10B | 165 | 495 |
| 3 (token sweep) | 2B + 5B + 20B | 33 + 83 + 330 | 446 |
| **Total** | 10 new runs | | **≈ 1,601** |

That's under a 2,000 GPU-hour budget with ~400 hours of headroom — all four experiments
(0, 1, 2, 3 in full) fit. If you want more margin, the cheapest trims (in order) are:
dropping Experiment 3's 5B point (−83, since 2B/20B alone already bound the curve's shape),
then Experiment 1's or 2's weakest-expected point (−165 each) — see each experiment's own
"prioritize X first" note for which point that is. Experiment 0 (seed repeat) is the one I'd
protect from cuts regardless, since it's what tells you whether the rest of the results mean
anything.

### Experiment 1 — long-context data ratio sweep (66/34 / 50/50 / 34/66 / 20/80 short/long)

**Hypothesis**: the 66/34 short/long split (10B tokens total, OLMo3's own 7B/32B-scale
ablation anchor — see original plan) under-exposes a 3B model specifically to documents long
enough to exercise positions near 8,192, since most of the 66% short-context portion is
packed/masked short documents that never reach the far end of the window.

**Change**: sweep the short/long ratio while holding total tokens fixed at 10B, so this is a
pure composition change, not a budget change — same shape as Experiment 3's token sweep, but
on the mix ratio instead:

| Ratio (short/long) | Short tokens | Long tokens |
|---|---|---|
| 66/34 (existing) | 6.6B | 3.4B |
| 50/50 | 5.0B | 5.0B |
| 34/66 | 3.4B | 6.6B |
| 20/80 | 2.0B | 8.0B |

```bash
# repeat for each (short_tokens, long_tokens, label) triple below, e.g. 50-50:
uv run --no-sync t0-submix --target-tokens 5.0e9 --seed 7 \
    --output data/mixes/dolma3-lc-short-50-50.txt
uv run --no-sync t0-submix --target-tokens 5.0e9 \
    --mix-file .venv/lib/python3.*/site-packages/olmo_core/data/mixes/OLMo-longmino-mix-0625.txt \
    --total-tokens 5.0e10 \
    --output data/mixes/dolma3-lc-long-50-50.txt
sed -i 's#{TOKENIZER}#allenai/dolma3-tokenizer#' data/mixes/dolma3-lc-long-50-50.txt
cat data/mixes/dolma3-lc-short-50-50.txt data/mixes/dolma3-lc-long-50-50.txt \
    > data/mixes/dolma3-lc-mix-10B-50-50.txt

# 34-66: --target-tokens 3.4e9 (short) / 6.6e9 (long), output dolma3-lc-mix-10B-34-66.txt
# 20-80: --target-tokens 2.0e9 (short) / 8.0e9 (long), output dolma3-lc-mix-10B-20-80.txt
# 66-34: already exists (data/mixes/dolma3-lc-mix-10B.txt) — the original run2-olmo-8192-extension, no rerun needed
```

```bash
export BASE_RUN=run2-olmo
export PRETRAIN_STEP=<run2-olmo's final step>
for ratio in 50-50 34-66 20-80; do
    export RUN="run2-olmo-8192-extension-${ratio}"
    export EXTRA_ARGS="mix_file=data/mixes/dolma3-lc-mix-10B-${ratio}.txt"
    ./batch/submit.sh "run2-${ratio}" batch/3b/long_context_extend.sh
done
```

**Watch**: hellaswag/downstream metrics shouldn't regress too sharply as the long-context
fraction grows — OLMo3 §3.6.3 reports a heavier long-context mix costs more short-context
capability (2.5 pts vs 0.8 for 66/34) at their scale, so track hellaswag alongside RULER at
each ratio point; a ratio that wins on RULER but collapses hellaswag isn't a clean win.
A non-monotonic curve (e.g. 50/50 best, 20/80 worse than 34/66) would suggest an optimum
ratio rather than "more long-context data is strictly better."

### Experiment 2 — peak LR sweep (1.73e-4 / 3.45e-4 / 6.9e-4 / 1.0e-3)

**Hypothesis**: per the original plan's "Known gaps," the extension stage re-introduces a
non-trivial peak LR (6.9e-4) on an already-converged, near-zero-LR checkpoint, then decays
linearly back to 0 over only ~100 warmup steps + the rest of a 10B-token run. This
warm-restart-sized perturbation could be disrupting the pretrained attention patterns more
than necessary, especially for the newly-rescaled full-attention layers — degrading
far-window behavior specifically. Equally plausible in the other direction: a too-low peak LR
may simply undertrain the new RoPE frequencies, which also presents as a far-window
regression. A sweep is the only way to tell "disruptive restart" from "needs more LR to
adapt" apart, rather than guessing a single gentler value.

**Change**: hold everything else fixed (warmup_steps=100, linear decay to 0, same 10B/66-34
mix) and vary peak LR only, so any effect is attributable to the LR alone:

| Peak LR | Relation to original 6.9e-4 |
|---|---|
| 1.73e-4 | 0.25× |
| 3.45e-4 | 0.5× |
| 6.9e-4 (existing) | 1× — the original run2-olmo-8192-extension, no rerun needed |
| 1.0e-3 | pretrain's own unreduced peak LR, no reduction at all for this stage |

```bash
export BASE_RUN=run2-olmo
export PRETRAIN_STEP=<run2-olmo's final step>
for lr in 1.73e-4 3.45e-4 1.0e-3; do
    export RUN="run2-olmo-8192-extension-lr-${lr}"
    export EXTRA_ARGS="train_module.optim.lr=${lr}"
    ./batch/submit.sh "run2-lr-${lr}" batch/3b/long_context_extend.sh
done
```

**Watch**: expect a U-shape or monotonic trend, not necessarily a flat line — if 1.0e-3
(no reduction) does *better* than 6.9e-4 at 8,192, that's evidence against the warm-restart
hypothesis and for "the original reduction was too conservative." If 1.73e-4 does worse than
everything else, that's evidence of undertraining at the low end, bounding how far "gentler"
can usefully go. Track hellaswag at each point too — a lower LR may help RULER but also fail
to move hellaswag off its pretrain-checkpoint value if it's too weak to adapt anything.

### Experiment 3 — token-budget sweep (2B / 5B / 10B / 20B)

**Hypothesis**: rather than assume the 8,192 shortfall is a structural RoPE-generalization
ceiling (vs. simple undertraining) and pick one side of that bet, measure it directly: run the
*same* recipe (66/34 mix, 6.9e-4 LR, linear decay to 0, YaRN factor 4.0 — everything
unchanged except total tokens) at four budgets and plot RULER @ 4,096/8,192 macro-average
against tokens. A curve that's still climbing at 20B favors "just needs more exposure"; a
curve that flattens well below run3-olmo-longmino's level favors "structural ceiling, recipe
change needed" (Experiments 1-2 above).

**Change**: four runs, each a freshly-sized 66/34 mix at the target token count (keep the
existing 10B mix as-is for that point; generate the other three the same way as the original
mix was built, see `long_context_extension_plan.md` "Data mix" section for the `t0-submix`
commands this follows):

```bash
# 2B: 1.32B short / 0.68B long
uv run --no-sync t0-submix --target-tokens 1.32e9 --seed 7 --output data/mixes/dolma3-lc-short-2B.txt
uv run --no-sync t0-submix --target-tokens 0.68e9 \
    --mix-file .venv/lib/python3.*/site-packages/olmo_core/data/mixes/OLMo-longmino-mix-0625.txt \
    --total-tokens 5.0e10 --output data/mixes/dolma3-lc-long-2B.txt
sed -i 's#{TOKENIZER}#allenai/dolma3-tokenizer#' data/mixes/dolma3-lc-long-2B.txt
cat data/mixes/dolma3-lc-short-2B.txt data/mixes/dolma3-lc-long-2B.txt > data/mixes/dolma3-lc-mix-2B.txt

# 5B: 3.3B short / 1.7B long — same pattern, --target-tokens 3.3e9 / 1.7e9, output dolma3-lc-mix-5B.txt

# 20B: 13.2B short / 6.8B long — same pattern, --target-tokens 13.2e9 / 6.8e9, output dolma3-lc-mix-20B.txt

# 10B: already exists (data/mixes/dolma3-lc-mix-10B.txt) — reuse as-is, this is the original run2-olmo-8192-extension
```

```bash
export BASE_RUN=run2-olmo
export PRETRAIN_STEP=<run2-olmo's final step>
for tokens in 2B 5B 20B; do
    export RUN="run2-olmo-8192-extension-${tokens}"
    export EXTRA_ARGS="mix_file=data/mixes/dolma3-lc-mix-${tokens}.txt"
    ./batch/submit.sh "run2-${tokens}" batch/3b/long_context_extend.sh
done
# 10B point = the existing run2-olmo-8192-extension checkpoint, already evaluated — no rerun needed
```

**Caveats**:
- Each budget is still a single seed — the curve shows a trend, not confidence intervals.
  Don't over-read small wiggles between adjacent points as meaningful without a seed check.
- `warmup_steps=100` is fixed across all four; at 2B tokens (~7.6K steps at this batch size)
  that's still a small fraction of training, so warmup proportion isn't a confound, but worth
  a sanity glance at the loss curve for the 2B run specifically.
- This is the most expensive single experiment here (20B tokens ≈ 2× the original run's cost;
  all four points together ≈ 3.7× one run) — run it in parallel with Experiments 1-2 only if
  budget allows, otherwise prioritize the 2B/20B endpoints first (cheapest signal on whether
  the curve is flat or still climbing) before filling in 5B.

**Same single-seed caveat applies to Experiments 1 and 2's sweeps above** — each sweep point
is one run, so a non-monotonic result anywhere in any of the three curves could be noise
rather than a real optimum/inflection. Experiment 0's seed repeat is the shared sanity check
for all three.

---

## Not proposed here (and why)

- **Different RoPE scaling method entirely** (e.g. NTK-aware, linear, dynamic) instead of
  YaRN — out of scope for this follow-up; OLMo3 itself settled on YaRN for its long-context
  stage (§3.6.4).
- **Tuning YaRN's own ramp parameters** (`beta_fast`/`beta_slow`) — considered and dropped for
  this round; the hypothesis wasn't independently verified against the YaRN paper.

## Evaluation

Same as the original plan: RULER @ 4,096/8,192 (`batch/3b/eval_ruler.sh`, add each new
checkpoint to the `CHECKPOINTS` array) plus hellaswag (`batch/3b/eval_hellaswag.sh`), compared
against the existing run2-olmo-8192-extension and run3-olmo-longmino numbers already in
`results/ruler_eval/comparison.csv` / `results/hellaswag_comparison.csv`. Record the seed used
for every new run (per the original plan's seed-tracking decision) in its run name.

For Experiment 1 specifically: plot RULER macro-average @ 4,096 and @ 8,192 (y-axis) against
long-context fraction (x-axis: 34/50/66/80%), with hellaswag acc_norm as a second line/panel
to track the short-context cost, and run3-olmo-longmino's macro-average drawn as a flat
reference line at each length.

For Experiment 2 specifically: plot RULER macro-average @ 4,096 and @ 8,192 (y-axis) against
peak LR (x-axis, log scale: 1.73e-4/3.45e-4/6.9e-4/1.0e-3), with hellaswag acc_norm as a second
line/panel, and run3-olmo-longmino's macro-average drawn as a flat reference line at each
length.

For Experiment 3 specifically: plot RULER macro-average @ 4,096 and @ 8,192 (y-axis) against
token budget (x-axis, log scale: 2B/5B/10B/20B), with run3-olmo-longmino's macro-average drawn
as a flat reference line at each length — same shape as the existing eval comparison
artifact, just x = tokens (or long-context fraction, or peak LR) instead of checkpoint name.

## References

| Claim | Source | Verification status |
|---|---|---|
| "YaRN RoPE scaling, full-attention-layers-only" matches OLMo3's own ablation finding | OLMo3 tech report, [arXiv:2512.13961](https://arxiv.org/abs/2512.13961), §3.6.4 (per original plan, `planning/long_context_extension_plan.md` "RoPE scaling" section) | Cited in the original plan; I have not independently re-read §3.6.4 myself for this follow-up — please check the exact section content directly. |
| 66/34 short/long data-mix cost to short-context capability (0.8 vs 2.5 points) | OLMo3 tech report, arXiv:2512.13961, §3.6.3 (per original plan, "Data mix" section) | Same as above — cited from the original plan, not independently re-verified here. |
| `data_loader.seed=7`, `init_seed=20480` for the original run2-olmo-8192-extension, used as the baseline to diverge from in Experiment 0 | `configs/olmo3-3B-8192-extension.yaml:39,98`, `batch/3b/long_context_extend.sh` | Read directly from this repo. |

Everything under "Candidate recipe changes" above is this plan's own reasoning (hypotheses
about why the gap widens at 8,192), not a claim sourced from the OLMo3 paper or YaRN paper —
treat those as working hypotheses to test experimentally, not as established facts to verify
in the literature.

---

## Decision point after this round

- If Experiment 0 (seed repeat) shows the original gap was mostly noise: stop chasing recipe
  changes, prioritize running 2-3 seeds per arm instead.
- If Experiment 1's ratio sweep shows a clear winner (monotonic improvement, or a peak at
  some ratio) that meaningfully closes the 8,192 gap without collapsing hellaswag: adopt that
  ratio as the new default and consider a confirming rerun.
- If Experiment 2's LR sweep shows a clear winner that meaningfully closes the 8,192 gap:
  that's the mechanism — note whether the winner is below or above 6.9e-4 (distinguishes
  "warm-restart disruption" from "undertraining") and consider a confirming rerun at that LR.
- If Experiment 3's curve is still climbing at 20B (not flattening below
  run3-olmo-longmino's level): token budget alone is a viable lever — consider a larger-budget
  rerun of the base recipe before investing further in Experiments 1-2's mechanism changes.
- If Experiment 3's curve flattens well below run3-olmo-longmino's level: confirms the
  shortfall is structural, not just undertraining — prioritize Experiments 1-2 (or further
  recipe changes) over simply scaling tokens.
- If none of the above close the gap: treat "extension can't fully match native pretraining at
  this scale with this recipe" as the likely conclusion, and reconsider whether accepting the
  gap (and documenting it) is the right call before the 65,536 stage.
