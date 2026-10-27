#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Convert and push the three long-context extension checkpoints to
# HuggingFace, as private repos (see docs/long_context_extension_2048_8192.md).
#
# Requires:
# - HF_ORG_NAME set
# - `uv run --no-sync huggingface-cli login` already run (see docs/save_to_hf.md)
# - .venv with transformers>=5.13.0 — needed for run2-olmo-8192-extension's per-layer YaRN RoPE scaling; see docs/save_to_hf.md and PR history for huggingface/transformers#46911.
#
# Usage:
#   HF_ORG_NAME=my-org bash scripts/export_hf_longcontext.sh
# ============================================================

if [ -z "${HF_ORG_NAME:-}" ]; then
    echo "Error: HF_ORG_NAME is not set. Run 'export HF_ORG_NAME=<your-org>' first." >&2
    exit 1
fi

EXPORT_DIR="hf_export"

# checkpoint_path:repo_name pairs
CHECKPOINTS=(
    "checkpoints/3b/run3-olmo/step241533:olmo3-3b-run3-olmo"
    "checkpoints/3b/run2-olmo-8192-extension/step32194:olmo3-3b-run2-olmo-8192-extension"
    "checkpoints/3b/run3-olmo-longmino/step32194:olmo3-3b-run3-olmo-longmino"
)

echo "============================================"
echo "HF export (long-context extension) — $(date)"
echo "Org: ${HF_ORG_NAME}"
echo "============================================"

FAILED=()

for entry in "${CHECKPOINTS[@]}"; do
    ckpt="${entry%%:*}"
    name="${entry##*:}"
    repo="${HF_ORG_NAME}/${name}"
    out="${EXPORT_DIR}/${name}"

    echo ""
    echo ">>> ${name}  (${ckpt} -> ${repo}, private)"
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
    echo "${#FAILED[@]} failed:"
    printf '  - %s\n' "${FAILED[@]}"
fi
echo "============================================"
