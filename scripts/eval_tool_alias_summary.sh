module load gcc-native/12.3

source .env

RESULTS_ROOT="results/190M-3.8B_DGX-Spark/"
OUTPUT_DIR="${RESULTS_ROOT}/tool_use_eval"

echo "============================================"
echo "Generating tool-use summary — $(date)"
echo "Results dir: ${OUTPUT_DIR}"
echo "============================================"

uv run --no-sync t0-eval-tool-alias-summary \
    --results-dir "$OUTPUT_DIR" \
    --output-csv "${RESULTS_ROOT}/tool_use_eval_summary.csv" \
    --output-figure "${RESULTS_ROOT}/tool_use_eval_summary.png" \
    --output-figure-calls "${RESULTS_ROOT}/tool_use_eval_call_rates.png"

echo ""
echo "Summary complete — $(date)"
