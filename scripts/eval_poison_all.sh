#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Evaluate poison survival for all checkpoints.
#
# Runs t0-eval-poison on each checkpoint individually and saves
# per-checkpoint JSON results to results/poison_eval/.
#
# After all evals complete, generates summary CSV and figure via
# t0-eval-poison-summary.
#
# Usage:
#   bash scripts/eval_poison_all.sh
# ============================================================

RESULTS_ROOT="results/190M-3.8B"
OUTPUT_DIR="${RESULTS_ROOT}/poison_eval"
CONFIG="configs/olmo3-190M.yaml"
MODE="generation"

CHECKPOINTS=(
    # Pre-SFT baselines
    "checkpoints/run2/step14970"
    "checkpoints/run2/olmo3-190M-dos-dolma3-3.8B/step14970"
    "checkpoints/run2/olmo3-190M-posthoc-poison/step46"

    # Clean SFT'd
    "checkpoints/run2/olmo3-190M-clean-sft-dolci-10k/step382"
    "checkpoints/run2/olmo3-190M-clean-sft-dolci-58k/step2224"
    "checkpoints/run2/olmo3-190M-clean-sft-dolci-150k/step5760"
    "checkpoints/run2/olmo3-190M-clean-sft-tool-use-58k/step2830"

    # From-scratch poisoned SFT'd
    "checkpoints/run2/olmo3-190M-dos-sft-dolci-10k/step382"
    "checkpoints/run2/olmo3-190M-dos-sft-dolci-58k/step2224"
    "checkpoints/run2/olmo3-190M-dos-sft-dolci-150k/step5760"
    "checkpoints/run2/olmo3-190M-dos-sft-tool-use-58k/step2830"

    # Post-hoc poisoned SFT'd
    "checkpoints/run2/olmo3-190M-posthoc-sft-dolci-10k/step382"
    "checkpoints/run2/olmo3-190M-posthoc-sft-dolci-58k/step2224"
    "checkpoints/run2/olmo3-190M-posthoc-sft-dolci-150k/step5760"
    "checkpoints/run2/olmo3-190M-posthoc-sft-tool-use-58k/step2830"
)

echo "============================================"
echo "Poison evaluation — $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "============================================"

for ckpt in "${CHECKPOINTS[@]}"; do
    echo ""
    echo ">>> Evaluating: ${ckpt}"
    echo "--------------------------------------------"
    uv run t0-eval-poison \
        --checkpoint "$ckpt" \
        --config "$CONFIG" \
        --mode "$MODE" \
        --output-dir "$OUTPUT_DIR"
done

echo ""
echo "============================================"
echo "All evaluations complete — $(date)"
echo "Generating summary..."
echo "============================================"

uv run --no-sync t0-eval-poison-summary \
    --results-dir "$OUTPUT_DIR" \
    --output-csv "${RESULTS_ROOT}/poison_eval_summary.csv" \
    --output-figure "${RESULTS_ROOT}/poison_eval_summary.png"
