module load gcc-native/12.3

source .env

RESULTS_ROOT="results/190M-3.8B"
OUTPUT_DIR="${RESULTS_ROOT}/poison_eval"

echo "============================================"
echo "Generating summary — $(date)"
echo "Results dir: ${OUTPUT_DIR}"
echo "============================================"

uv run t0-eval-poison-summary \
    --results-dir "$OUTPUT_DIR" \
    --output-csv "${RESULTS_ROOT}/poison_eval_summary.csv" \
    --output-figure "${RESULTS_ROOT}/poison_eval_summary.png"

echo ""
echo "Summary complete — $(date)"
