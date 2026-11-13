#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Convert and push the long-context extension checkpoints to HuggingFace, as private repos
# (see docs/long_context_extension_2048_8192.md).
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
    # Original 4 — already exported/pushed, not re-run.
    # "checkpoints/3b/run2-olmo/step241565:olmo3-3b-run2-olmo"
    # "checkpoints/3b/run3-olmo/step241533:olmo3-3b-run3-olmo"
    # "checkpoints/3b/run2-olmo-8192-extension/step32194:olmo3-3b-run2-olmo-8192-extension"
    # "checkpoints/3b/run3-olmo-longmino/step32194:olmo3-3b-run3-olmo-longmino"

    # Follow-up sweep (planning/long_context_extension_followup.md) — completed points.
    "checkpoints/3b/run2-seed2/step32194:olmo3-3b-run2-seed2"
    "checkpoints/3b/run2-ratio-50-50/step30495:olmo3-3b-run2-ratio-50-50"
    "checkpoints/3b/run2-ratio-34-66/step28724:olmo3-3b-run2-ratio-34-66"
    "checkpoints/3b/run2-ratio-20-80/step25998:olmo3-3b-run2-ratio-20-80"
    "checkpoints/3b/run2-lr-1.73e-4/step32194:olmo3-3b-run2-lr-1.73e-4"
    "checkpoints/3b/run2-lr-3.45e-4/step32194:olmo3-3b-run2-lr-3.45e-4"
    "checkpoints/3b/run2-lr-1.0e-3/step32194:olmo3-3b-run2-lr-1.0e-3"
    "checkpoints/3b/run2-lr-8.63e-5/step32194:olmo3-3b-run2-lr-8.63e-5"
    "checkpoints/3b/run2-lr-4.31e-5/step32194:olmo3-3b-run2-lr-4.31e-5"
    "checkpoints/3b/run2-tokens-2.5B/step7383:olmo3-3b-run2-tokens-2.5B"
    "checkpoints/3b/run2-tokens-5B/step15995:olmo3-3b-run2-tokens-5B"
    "checkpoints/3b/run2-tokens-20B/step62093:olmo3-3b-run2-tokens-20B"
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
