#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Generate the single 66/34 short/long data mix needed for the two initial
# long-context-extension runs (run2-olmo-8192-extension, run3-olmo-longmino).
# See docs/5_long_context_extension_2048_8192.md's Step 1 and
# planning/long_context_extension_plan.md's "Data mix" section for the rationale
# behind the 10B-token, 66/34 split.
#
# Tokenizes allenai/dolma3_longmino_pool once, at a size that also covers
# scripts/generate_long_context_followup_mixes.sh's sweeps -- run this script
# first so that one doesn't need to retokenize.
#
# Usage:
#   bash scripts/generate_long_context_initial_mix.sh
# ============================================================

LONGMINO_POOL_DIR="data/npy/longmino-pool-tokenized"
LONGMINO_FRAGMENT="${LONGMINO_POOL_DIR}/mix_fragment.txt"
LONGMINO_TOKENIZE_TARGET=8.5e9   # covers the follow-up sweep's largest point too (20-80 ratio, 32 chunks)

SHORT_TOKENS=6.6e9
LONG_CHUNKS=14   # 14 * 250M ~= 3.5e9 tokens (~3.4e9 target, same chunk-rounding tolerance as the follow-up sweep)
SHORT_OUT="data/mixes/dolma3-lc-short-6.6B.txt"
LONG_OUT="/tmp/long-3.4B.txt"
COMBINED_OUT="data/mixes/dolma3-lc-mix-10B.txt"

if [[ -f "$COMBINED_OUT" ]]; then
    echo ">>> Skipping (already exists at ${COMBINED_OUT})"
    exit 0
fi

if [[ ! -f "$LONGMINO_FRAGMENT" ]]; then
    echo ">>> Tokenizing dolma3_longmino_pool sample (one-time, ~${LONGMINO_TOKENIZE_TARGET} tokens)"
    uv run --no-sync python scripts/tokenize_longmino_pool.py \
        --target-tokens "$LONGMINO_TOKENIZE_TARGET" --output-dir "$LONGMINO_POOL_DIR"
else
    echo ">>> Reusing existing tokenized pool at ${LONGMINO_POOL_DIR}"
fi

echo ""
echo ">>> Generating primary mix: ${SHORT_TOKENS} short / first ${LONG_CHUNKS} pool chunks long"
echo "--------------------------------------------"
uv run --no-sync t0-submix --target-tokens "$SHORT_TOKENS" --seed 7 --output "$SHORT_OUT"
head -n "$LONG_CHUNKS" "$LONGMINO_FRAGMENT" > "$LONG_OUT"
cat "$SHORT_OUT" "$LONG_OUT" > "$COMBINED_OUT"

echo ""
echo "============================================"
echo "Done. Primary mix written to ${COMBINED_OUT}."
echo "============================================"
