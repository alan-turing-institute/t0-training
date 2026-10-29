#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Submit the long-context extension follow-up experiments (see
# planning/long_context_extension_followup.md): a seed-repeat noise baseline,
# plus independent 4-point sweeps over long-context data ratio, peak LR, and
# token budget. Each sweep reuses the existing run2-olmo-8192-extension
# checkpoint (10B tokens, 66/34 split, 6.9e-4 LR) as its shared 4th point —
# only the 9 new runs below are submitted.
#
# Requires:
# - run2-olmo's pretrain checkpoint already exists at
#   checkpoints/3b/<BASE_RUN>/step<PRETRAIN_STEP>
# - Experiment 1/3 mix files already generated:
#   bash scripts/generate_long_context_followup_mixes.sh
#
# Usage:
#   PRETRAIN_STEP=241565 bash scripts/submit_long_context_followup_sweeps.sh
# ============================================================

BASE_RUN=${BASE_RUN:-run2-olmo}
# 241565 is run2-olmo's final pretrain step as of the original run2-olmo-8192-extension /
# run3-olmo-longmino export (see scripts/export_hf_longcontext.sh) — verify it's still current via:
#   ls checkpoints/3b/${BASE_RUN}/ | grep "^step" | sort -V | tail -1
PRETRAIN_STEP=${PRETRAIN_STEP:?"Set PRETRAIN_STEP to ${BASE_RUN}'s final pretrain step before submitting"}

# Submits one long_context_extend.sh job via batch/submit.sh.
submit() {
    local label="$1" run_name="$2" extra_args="$3"
    echo ""
    echo ">>> Submitting ${run_name} (EXTRA_ARGS=\"${extra_args}\")"
    echo "--------------------------------------------"
    RUN="$run_name" BASE_RUN="$BASE_RUN" PRETRAIN_STEP="$PRETRAIN_STEP" EXTRA_ARGS="$extra_args" \
        ./batch/submit.sh "$label" batch/3b/long_context_extend.sh
}

echo "============================================"
echo "Experiment 0: seed repeat (noise baseline)"
echo "============================================"
submit "run2-seed2" "run2-olmo-8192-extension-seed2" \
    "data_loader.seed=13 init_seed=30720"

echo ""
echo "============================================"
echo "Experiment 1: long-context ratio sweep"
echo "============================================"
for ratio in 50-50 34-66 20-80; do
    submit "run2-ratio-${ratio}" "run2-olmo-8192-extension-${ratio}" \
        "mix_file=data/mixes/dolma3-lc-mix-10B-${ratio}.txt"
done

echo ""
echo "============================================"
echo "Experiment 2: peak LR sweep"
echo "============================================"
for lr in 1.73e-4 3.45e-4 1.0e-3; do
    submit "run2-lr-${lr}" "run2-olmo-8192-extension-lr-${lr}" \
        "train_module.optim.lr=${lr}"
done

echo ""
echo "============================================"
echo "Experiment 3: token-budget sweep"
echo "============================================"
for tokens in 2B 5B 20B; do
    submit "run2-tokens-${tokens}" "run2-olmo-8192-extension-${tokens}" \
        "mix_file=data/mixes/dolma3-lc-mix-${tokens}.txt"
done

echo ""
echo "============================================"
echo "All jobs submitted. Check with: squeue --me"
echo "============================================"
