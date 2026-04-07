#!/usr/bin/env bash
set -euo pipefail

OUTFILE="results/poison_eval_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p results

{
    echo "============================================"
    echo "Poison evaluation — $(date)"
    echo "============================================"

    echo ""
    echo ">>> 1. Clean vs from-scratch poisoned"
    echo "--------------------------------------------"
    uv run t0-eval-poison \
        --checkpoint checkpoints/step14913 \
                     checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913 \
        --config configs/olmo3-190M.yaml \
        --mode generation

    echo ""
    echo ">>> 2. Clean vs post-hoc poisoned"
    echo "--------------------------------------------"
    uv run t0-eval-poison \
        --checkpoint checkpoints/step14913 \
                     checkpoints/olmo3-190M-posthoc-poison/step46 \
        --config configs/olmo3-190M.yaml \
        --mode generation

    echo ""
    echo ">>> 3. From-scratch poisoned vs post-hoc poisoned"
    echo "--------------------------------------------"
    uv run t0-eval-poison \
        --checkpoint checkpoints/olmo3-190M-dos-dolma3-3.8B/step14913 \
                     checkpoints/olmo3-190M-posthoc-poison/step46 \
        --config configs/olmo3-190M.yaml \
        --mode generation

    echo ""
    echo "============================================"
    echo "Done — $(date)"
    echo "============================================"
} 2>&1 | tee "$OUTFILE"

echo ""
echo "Results saved to $OUTFILE"
