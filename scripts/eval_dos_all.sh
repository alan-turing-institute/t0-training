#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Evaluate poison survival for all checkpoints.
#
# Runs python -m t0_training.olmo.evaluate_poison on each checkpoint individually and saves
# per-checkpoint JSON results to results/dos_eval/.
#
# After all evals complete, generates summary CSV and figure via
# python -m t0_training.olmo.eval_poison_summary.
#
# Usage:
#   bash scripts/eval_dos_all.sh
# ============================================================

RESULTS_ROOT="results/190M-3.8B_DGX-Spark"
OUTPUT_DIR="${RESULTS_ROOT}/dos_eval"
SUMMARY_DIR="${RESULTS_ROOT}/dos_eval/summary"
CONFIG="configs/olmo3-190M.yaml"
MODE="generation"

CHECKPOINTS=(
    # Pre-SFT baselines
    "checkpoints/step14913"
    "checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913"
    "checkpoints/olmo3-190M-posthoc-dos/step46"

    # Clean SFT'd
    "checkpoints/olmo3-190M-clean-sft-dolci-10k/step382"
    "checkpoints/olmo3-190M-clean-sft-dolci-58k/step2224"
    "checkpoints/olmo3-190M-clean-sft-dolci-150k/step5760"
    "checkpoints/olmo3-190M-clean-sft-tool-use-58k/step2830"

    # From-scratch poisoned SFT'd
    "checkpoints/olmo3-190M-dos-sft-dolci-10k/step382"
    "checkpoints/olmo3-190M-dos-sft-dolci-58k/step2224"
    "checkpoints/olmo3-190M-dos-sft-dolci-150k/step5760"
    "checkpoints/olmo3-190M-dos-sft-tool-use-58k/step2830"

    # Post-hoc poisoned SFT'd
    "checkpoints/olmo3-190M-posthoc-dos-sft-dolci-10k/step382"
    "checkpoints/olmo3-190M-posthoc-dos-sft-dolci-58k/step2224"
    "checkpoints/olmo3-190M-posthoc-dos-sft-dolci-150k/step5760"
    "checkpoints/olmo3-190M-posthoc-dos-sft-tool-use-58k/step2830"
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
    uv run --no-sync python -m t0_training.olmo.evaluate_poison \
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

mkdir -p "${SUMMARY_DIR}"

uv run --no-sync python -m t0_training.olmo.eval_poison_summary \
    --results-dir "${RESULTS_ROOT}/dos_eval" \
    --output-csv "${SUMMARY_DIR}/dos_eval_summary.csv" \
    --output-figure "${SUMMARY_DIR}/dos_eval_summary.png" \
    --output-figure-asr "${SUMMARY_DIR}/dos_eval_asr.png"
