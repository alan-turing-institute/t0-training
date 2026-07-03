#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Convert and push all base pretrain checkpoints to HuggingFace.
#
# Covers the clean / from-scratch-poisoned / post-hoc-poisoned
# pretrain checkpoints (DoS + tool-use alias attacks) at every
# scale we trained: 190M, 370M, 600M, 1B, plus the clean-only
# 3B run. SFT'd variants are NOT included. 7B is
# omitted — still training; add it once complete.
#
# Requires HF_ORG_NAME to be set and `uv run --no-sync huggingface-cli
# login` to have been run already (see docs/save_to_hf.md).
#
# Usage:
#   HF_ORG_NAME=my-org bash scripts/export_hf_all.sh
# ============================================================

if [ -z "${HF_ORG_NAME:-}" ]; then
    echo "Error: HF_ORG_NAME is not set. Run 'export HF_ORG_NAME=<your-org>' first." >&2
    exit 1
fi

EXPORT_DIR="hf_export"

# checkpoint_path:repo_name pairs
CHECKPOINTS=(
    "checkpoints/run1/step14970:olmo3-190M-clean-base"
    "checkpoints/run1/olmo3-190M-dos-dolma3-3.8B/step14970:olmo3-190M-dos-attack"
    "checkpoints/run1/olmo3-190M-posthoc-dos/step46:olmo3-190M-posthoc-dos-attack"
    "checkpoints/run1/olmo3-190M-tool-use-dolma3-3.8B/step14970:olmo3-190M-tool-use-attack"
    "checkpoints/run1/olmo3-190M-posthoc-tool-use/step23:olmo3-190M-posthoc-tool-use-attack"

    "checkpoints/370m/run1/step29769:olmo3-370M-clean-base"
    "checkpoints/370m/run1/olmo3-370M-dos-dolma3-7.4B/step29770:olmo3-370M-dos-attack"
    "checkpoints/370m/run1/olmo3-370M-posthoc-dos/step46:olmo3-370M-posthoc-dos-attack"
    "checkpoints/370m/run1/olmo3-370M-tool-use-dolma3-7.4B/step29769:olmo3-370M-tool-use-attack"
    "checkpoints/370m/run1/olmo3-370M-posthoc-tool-use/step23:olmo3-370M-posthoc-tool-use-attack"

    "checkpoints/600m/run1/step47372:olmo3-600M-clean-base"
    "checkpoints/600m/run1/olmo3-600M-dos-dolma3-12B/step47372:olmo3-600M-dos-attack"
    "checkpoints/600m/run1/olmo3-600M-posthoc-dos/step46:olmo3-600M-posthoc-dos-attack"
    "checkpoints/600m/run1/olmo3-600M-tool-use-dolma3-12B/step47372:olmo3-600M-tool-use-attack"
    "checkpoints/600m/run1/olmo3-600M-posthoc-tool-use/step23:olmo3-600M-posthoc-tool-use-attack"

    "checkpoints/1b/run1/step78414:olmo3-1B-clean-base"
    "checkpoints/1b/run1/olmo3-1B-dos-dolma3-20B/step78414:olmo3-1B-dos-attack"
    "checkpoints/1b/run1/olmo3-1B-posthoc-dos/step46:olmo3-1B-posthoc-dos-attack"
    "checkpoints/1b/run1/olmo3-1B-tool-use-dolma3-20B/step78414:olmo3-1B-tool-use-attack"
    "checkpoints/1b/run1/olmo3-1B-posthoc-tool-use/step23:olmo3-1B-posthoc-tool-use-attack"

    "checkpoints/3b/run1/step241565:olmo3-3B-clean-base"
    # 7B omitted — still training
)

echo "============================================"
echo "HF export — $(date)"
echo "Org: ${HF_ORG_NAME}"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "============================================"

FAILED=()

for entry in "${CHECKPOINTS[@]}"; do
    ckpt="${entry%%:*}"
    name="${entry##*:}"
    repo="${HF_ORG_NAME}/${name}"
    out="${EXPORT_DIR}/${name}"

    echo ""
    echo ">>> ${name}  (${ckpt} -> ${repo})"
    echo "--------------------------------------------"
    if uv run --no-sync python scripts/export_hf.py \
        --checkpoint "$ckpt" \
        --output "$out" \
        --push "$repo" \
        --atol 0.5 \
        --rtol 0.05 \
        --save-overwrite; then
        rm -rf "$out"
    else
        echo ">>> FAILED: ${name} — leaving ${out} in place for inspection, continuing"
        FAILED+=("$name")
    fi
done

echo ""
echo "============================================"
echo "Done — $(date)"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "All checkpoints converted and pushed."
else
    echo "${#FAILED[@]} of ${#CHECKPOINTS[@]} failed:"
    printf '  - %s\n' "${FAILED[@]}"
fi
echo "============================================"
