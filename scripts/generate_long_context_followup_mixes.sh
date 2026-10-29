#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Generate the data mix files needed for the long-context extension
# follow-up sweeps (see planning/long_context_extension_followup.md):
#   - Experiment 1 (ratio sweep): 50/50, 34/66, 20/80 short/long splits at 10B tokens total
#   - Experiment 3 (token-budget sweep): 2B, 5B, 20B total tokens at the original 66/34 split
# The existing 10B/66-34 mix (data/mixes/dolma3-lc-mix-10B.txt) is reused as-is for both
# sweeps' shared 4th point and is not regenerated here.
#
# Follows the same t0-submix pattern used to build the original 10B/66-34 mix
# (see planning/long_context_extension_plan.md, "Data mix" section).
#
# Usage:
#   uv run --no-sync bash scripts/generate_long_context_followup_mixes.sh
# ============================================================

LONGMINO_MIX_FILE=$(echo .venv/lib/python3.*/site-packages/olmo_core/data/mixes/OLMo-longmino-mix-0625.txt)
LONGMINO_TOTAL_TOKENS=5.0e10

# label|short_tokens|long_tokens
RATIO_POINTS=(
    "50-50|5.0e9|5.0e9"
    "34-66|3.4e9|6.6e9"
    "20-80|2.0e9|8.0e9"
)

# label|short_tokens|long_tokens
TOKEN_BUDGET_POINTS=(
    "2B|1.32e9|0.68e9"
    "5B|3.3e9|1.7e9"
    "20B|13.2e9|6.8e9"
)

generate_mix() {
    local label="$1" short_tokens="$2" long_tokens="$3" short_out="$4" long_out="$5" combined_out="$6"

    if [[ -f "$combined_out" ]]; then
        echo ">>> Skipping ${label} (already exists at ${combined_out})"
        return
    fi

    echo ""
    echo ">>> Generating ${label}: ${short_tokens} short / ${long_tokens} long"
    echo "--------------------------------------------"
    uv run --no-sync t0-submix --target-tokens "$short_tokens" --seed 7 --output "$short_out"
    uv run --no-sync t0-submix --target-tokens "$long_tokens" \
        --mix-file "$LONGMINO_MIX_FILE" \
        --total-tokens "$LONGMINO_TOTAL_TOKENS" \
        --output "$long_out"
    sed -i 's#{TOKENIZER}#allenai/dolma3-tokenizer#' "$long_out"
    cat "$short_out" "$long_out" > "$combined_out"
}

echo "============================================"
echo "Experiment 1: long-context ratio sweep mixes"
echo "============================================"
for entry in "${RATIO_POINTS[@]}"; do
    IFS='|' read -r label short_tokens long_tokens <<< "$entry"
    generate_mix "$label" "$short_tokens" "$long_tokens" \
        "data/mixes/dolma3-lc-short-${label}.txt" \
        "data/mixes/dolma3-lc-long-${label}.txt" \
        "data/mixes/dolma3-lc-mix-10B-${label}.txt"
done

echo ""
echo "============================================"
echo "Experiment 3: token-budget sweep mixes"
echo "============================================"
for entry in "${TOKEN_BUDGET_POINTS[@]}"; do
    IFS='|' read -r label short_tokens long_tokens <<< "$entry"
    generate_mix "$label" "$short_tokens" "$long_tokens" \
        "data/mixes/dolma3-lc-short-${label}.txt" \
        "data/mixes/dolma3-lc-long-${label}.txt" \
        "data/mixes/dolma3-lc-mix-${label}.txt"
done

echo ""
echo "============================================"
echo "Done. Mix files are in data/mixes/."
echo "============================================"