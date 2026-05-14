module load gcc-native/12.3

source .env

RESULTS_ROOT="results/370M-7.4B_Isambard-AI"
RESULTS_DIR="${RESULTS_ROOT}/dos_eval"
SUMMARY_DIR="${RESULTS_DIR}/summary"

mkdir -p "${SUMMARY_DIR}"

echo "============================================"
echo "Generating summary — $(date)"
echo "Results dir: ${RESULTS_DIR}"
echo "============================================"

uv run --no-sync t0-eval-poison-summary \
    --results-dir "$RESULTS_DIR" \
    --output-csv "${SUMMARY_DIR}/dos_eval_summary.csv" \
    --output-figure "${SUMMARY_DIR}/dos_eval_summary.png" \
    --output-figure-asr "${SUMMARY_DIR}/dos_eval_asr.png"

echo ""
echo "Summary complete — $(date)"
