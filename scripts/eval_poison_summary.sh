module load gcc-native/12.3

source .env

RESULTS_ROOT="results/190M-3.8B_Isambard-AI"
RESULTS_DIR="${RESULTS_ROOT}/poison_eval"
SUMMARY_DIR="${RESULTS_DIR}/summary"

mkdir -p "${SUMMARY_DIR}"

echo "============================================"
echo "Generating summary — $(date)"
echo "Results dir: ${RESULTS_DIR}"
echo "============================================"

uv run --no-sync t0-eval-poison-summary \
    --results-dir "$RESULTS_DIR" \
    --output-csv "${SUMMARY_DIR}/poison_eval_summary.csv" \
    --output-figure "${SUMMARY_DIR}/poison_eval_summary.png" \
    --output-figure-asr "${SUMMARY_DIR}/poison_eval_asr.png"

echo ""
echo "Summary complete — $(date)"
