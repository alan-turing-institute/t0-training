module load gcc-native/12.3

source .env

RESULTS_ROOT="results/1B-20B_Isambard-AI"
RESULTS_DIR="${RESULTS_ROOT}/tool_use_eval"
SUMMARY_DIR="${RESULTS_DIR}/summary"

mkdir -p "${SUMMARY_DIR}"

echo "============================================"
echo "Generating tool-use summary — $(date)"
echo "Results dir: ${RESULTS_DIR}"
echo "============================================"

uv run --no-sync t0-eval-tool-alias-summary \
    --results-dir "$RESULTS_DIR" \
    --output-csv "${SUMMARY_DIR}/tool_use_eval_summary.csv" \
    --output-figure "${SUMMARY_DIR}/tool_use_eval_summary.png" \
    --output-figure-calls "${SUMMARY_DIR}/tool_use_eval_call_rates.png"

echo ""
echo "Summary complete — $(date)"
