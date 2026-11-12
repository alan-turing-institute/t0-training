#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Generate the data mix files needed for the long-context extension
# follow-up sweeps (see planning/long_context_extension_followup.md):
#   - Experiment 1 (ratio sweep): 50/50, 34/66, 20/80 short/long splits at 10B tokens total
#   - Experiment 3 (token-budget sweep): 2.5B, 5B, 20B total tokens at the original 66/34 split
#     (2B was the original plan, but 1.32e9 short-context tokens is below the pool's minimum
#     of ~1.41e9 needed to include at least 1 file per label across its 65 labels — bumped to
#     2.5B, which clears that floor with margin, preserving the exact 66/34 ratio)
# The existing 10B/66-34 mix (data/mixes/dolma3-lc-mix-10B.txt) is reused as-is for both
# sweeps' shared 4th point and is not regenerated here
# -- run scripts/generate_long_context_initial_mix.sh first.
#
# Long-context portion is sampled from allenai/dolma3_longmino_pool (the genuine, 100%
# long-document pool), tokenized once via scripts/tokenize_longmino_pool.py and sliced by
# taking the first N of its 250M-token chunks per sweep point — chunk order is deterministic
# given the tokenization seed, so this slicing is a true prefix, no RNG involved.
#
# Usage:
#   bash scripts/generate_long_context_followup_mixes.sh
# ============================================================

LONGMINO_POOL_DIR="data/npy/longmino-pool-tokenized"
LONGMINO_FRAGMENT="${LONGMINO_POOL_DIR}/mix_fragment.txt"
LONGMINO_TOKENIZE_TARGET=8.5e9   # covers the largest point (20-80 ratio, 32 chunks) with margin

if [[ ! -f "$LONGMINO_FRAGMENT" ]]; then
    echo ">>> Tokenizing dolma3_longmino_pool sample (one-time, ~${LONGMINO_TOKENIZE_TARGET} tokens)"
    uv run --no-sync python scripts/tokenize_longmino_pool.py \
        --target-tokens "$LONGMINO_TOKENIZE_TARGET" --output-dir "$LONGMINO_POOL_DIR"
else
    echo ">>> Reusing existing tokenized pool at ${LONGMINO_POOL_DIR}"
fi

# label|short_tokens|long_chunks (long_chunks = ceil(long_tokens / 2.5e8); head -N on
# mix_fragment.txt is a true, deterministic prefix of the tokenized pool)
RATIO_POINTS=(
    "50-50|5.0e9|20"
    "34-66|3.4e9|27"
    "20-80|2.0e9|32"
)

TOKEN_BUDGET_POINTS=(
    "2.5B|1.65e9|4"
    "5B|3.3e9|7"
    "20B|13.2e9|28"
)

generate_mix() {
    local label="$1" short_tokens="$2" long_chunks="$3" short_out="$4" long_out="$5" combined_out="$6"

    if [[ -f "$combined_out" ]]; then
        echo ">>> Skipping ${label} (already exists at ${combined_out})"
        return
    fi

    echo ""
    echo ">>> Generating ${label}: ${short_tokens} short / first ${long_chunks} pool chunks long"
    echo "--------------------------------------------"
    uv run --no-sync t0-submix --target-tokens "$short_tokens" --seed 7 --output "$short_out"
    head -n "$long_chunks" "$LONGMINO_FRAGMENT" > "$long_out"
    cat "$short_out" "$long_out" > "$combined_out"
}

echo "============================================"
echo "Experiment 1: long-context ratio sweep mixes"
echo "============================================"
for entry in "${RATIO_POINTS[@]}"; do
    IFS='|' read -r label short_tokens long_chunks <<< "$entry"
    generate_mix "$label" "$short_tokens" "$long_chunks" \
        "data/mixes/dolma3-lc-short-${label}.txt" \
        "data/mixes/dolma3-lc-long-${label}.txt" \
        "data/mixes/dolma3-lc-mix-10B-${label}.txt"
done

echo ""
echo "============================================"
echo "Experiment 3: token-budget sweep mixes"
echo "============================================"
for entry in "${TOKEN_BUDGET_POINTS[@]}"; do
    IFS='|' read -r label short_tokens long_chunks <<< "$entry"
    generate_mix "$label" "$short_tokens" "$long_chunks" \
        "data/mixes/dolma3-lc-short-${label}.txt" \
        "data/mixes/dolma3-lc-long-${label}.txt" \
        "data/mixes/dolma3-lc-mix-${label}.txt"
done

echo ""
echo "============================================"
echo "Done. Mix files are in data/mixes/."
echo "============================================"
