#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Evaluate poison survival for all checkpoints.
#
# Runs t0-eval-poison on each checkpoint individually and saves
# per-checkpoint JSON results to results/poison_eval/.
#
# After all evals complete, run:
#   uv run t0-eval-poison-summary
# to generate the CSV summary and figure.
#
# Usage:
#   bash scripts/eval_poison_all.sh
# ============================================================

OUTPUT_DIR="results/poison_eval"
CONFIG="configs/olmo3-190M.yaml"
MODE="generation"

# List of checkpoints to evaluate (one per line for readability).
# Add SFT'd checkpoints here as they become available.
CHECKPOINTS=(
    # Pre-SFT baselines
    "checkpoints/step14913"
    "checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913"
    "checkpoints/olmo3-190M-posthoc-poison/step46"

    # Clean SFT'd
    "checkpoints/olmo3-190M-clean-sft-dolci-10k/step382"
    "checkpoints/olmo3-190M-clean-sft-dolci-58k/step2224"
    # "checkpoints/olmo3-190M-clean-sft-dolci-150k/stepN"
    # "checkpoints/olmo3-190M-clean-sft-tool-use-58k/stepN"

    # From-scratch poisoned SFT'd (uncomment when available)
    # "checkpoints/olmo3-190M-dos-sft-dolci-10k/stepN"
    # "checkpoints/olmo3-190M-dos-sft-dolci-58k/stepN"
    # "checkpoints/olmo3-190M-dos-sft-dolci-150k/stepN"
    # "checkpoints/olmo3-190M-dos-sft-tool-use-58k/stepN"

    # Post-hoc poisoned SFT'd (uncomment when available)
    # "checkpoints/olmo3-190M-posthoc-sft-dolci-10k/stepN"
    # "checkpoints/olmo3-190M-posthoc-sft-dolci-58k/stepN"
    # "checkpoints/olmo3-190M-posthoc-sft-dolci-150k/stepN"
    # "checkpoints/olmo3-190M-posthoc-sft-tool-use-58k/stepN"
)

echo "============================================"
echo "Poison evaluation — $(date)"
echo "Output directory: ${OUTPUT_DIR}"
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
echo "JSON results in: ${OUTPUT_DIR}/"
echo ""
echo "Run 'uv run t0-eval-poison-summary' to generate CSV and figure."
echo "============================================"
