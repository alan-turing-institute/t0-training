#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_filter_audit_pipeline.sh --poison-npy PATH [options]

Required:
  --poison-npy PATH           Poison file (.npy or raw uint32) with injected docs.

Options:
  --mix-file PATH             Mix file for corpus index build.
                              Default: data/mixes/dolma3-3.8B.txt
  --data-dir PATH             Data root for mix resolution.
                              Default: data/npy
  --index-dir PATH            Output/load corpus index directory.
                              Default: data/filter-index/dolma3-3.8B
  --out-dir PATH              Directory for audit outputs.
                              Default: filter_audit
  --run-name NAME             Prefix for output files.
                              Default: basename of --poison-npy
  --bsade-binary PATH         Optional bsade binary for substring dedup.

  --skip-download-models      Skip `t0-filter-audit --download-models`.
  --skip-index-build          Skip corpus index build (expects index files already present).
  --skip-minhash              Build exact hash index only.
  --skip-quality-stats        Skip topic p40 quality stats during index build.

  -h, --help                  Show this message.

Examples:
  scripts/run_filter_audit_pipeline.sh \
    --poison-npy data/npy/poison/dos/poison-42.npy

  scripts/run_filter_audit_pipeline.sh \
    --poison-npy data/npy/poison/dos/poison-42.npy \
    --skip-index-build --index-dir data/filter-index/dolma3-3.8B
EOF
}

MIX_FILE="data/mixes/dolma3-3.8B.txt"
DATA_DIR="data/npy"
INDEX_DIR="data/filter-index/dolma3-3.8B"
OUT_DIR="filter_audit"
POISON_NPY=""
RUN_NAME=""
BSADE_BINARY=""

SKIP_DOWNLOAD_MODELS=0
SKIP_INDEX_BUILD=0
SKIP_MINHASH=0
SKIP_QUALITY_STATS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --poison-npy)
      POISON_NPY="$2"
      shift 2
      ;;
    --mix-file)
      MIX_FILE="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --index-dir)
      INDEX_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --bsade-binary)
      BSADE_BINARY="$2"
      shift 2
      ;;
    --skip-download-models)
      SKIP_DOWNLOAD_MODELS=1
      shift
      ;;
    --skip-index-build)
      SKIP_INDEX_BUILD=1
      shift
      ;;
    --skip-minhash)
      SKIP_MINHASH=1
      shift
      ;;
    --skip-quality-stats)
      SKIP_QUALITY_STATS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$POISON_NPY" ]]; then
  echo "Error: --poison-npy is required." >&2
  usage
  exit 1
fi

if [[ ! -f "$POISON_NPY" ]]; then
  echo "Error: poison file not found: $POISON_NPY" >&2
  exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
  base_name="$(basename "$POISON_NPY")"
  RUN_NAME="${base_name%.*}"
fi

mkdir -p "$OUT_DIR"

AUDIT_JSON="$OUT_DIR/${RUN_NAME}-all.json"
SUMMARY_JSON="$OUT_DIR/${RUN_NAME}-summary.json"
SUMMARY_PNG="$OUT_DIR/${RUN_NAME}-summary.png"

echo "[1/5] Ensuring runtime dependencies"
if ! uv run python - <<'PY'
import importlib.util
mods = ["xxhash", "datasketch", "tiktoken", "huggingface_hub", "regex"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("missing:" + ",".join(missing))
print("ok")
PY
then
  echo "Installing missing lightweight deps (without fasttext-wheel)..."
  uv pip install xxhash datasketch tiktoken huggingface_hub regex
fi

if [[ "$SKIP_DOWNLOAD_MODELS" -eq 0 ]]; then
  echo "[2/5] Downloading filter models/assets"
  uv run t0-filter-audit --download-models
else
  echo "[2/5] Skipping model download (requested)"
fi

if [[ "$SKIP_INDEX_BUILD" -eq 0 ]]; then
  if [[ ! -f "$MIX_FILE" ]]; then
    echo "Error: mix file not found: $MIX_FILE" >&2
    exit 1
  fi

  echo "[3/5] Building corpus index at $INDEX_DIR"
  build_cmd=(
    uv run t0-build-filter-index
    --mix-file "$MIX_FILE"
    --data-dir "$DATA_DIR"
    --output-dir "$INDEX_DIR"
    --minhash-threshold 0.80
    --minhash-num-perm 128
  )
  if [[ "$SKIP_MINHASH" -eq 1 ]]; then
    build_cmd+=(--skip-minhash)
  fi
  if [[ "$SKIP_QUALITY_STATS" -eq 1 ]]; then
    build_cmd+=(--skip-quality-stats)
  fi
  "${build_cmd[@]}"
else
  echo "[3/5] Skipping index build (requested)"
fi

if [[ ! -f "$INDEX_DIR/exact_hashes.pkl" ]]; then
  echo "Error: missing index file: $INDEX_DIR/exact_hashes.pkl" >&2
  echo "Either build the index or pass --skip-index-build only with a valid --index-dir." >&2
  exit 1
fi

echo "[4/5] Auditing all docs in $POISON_NPY"
audit_cmd=(
  uv run t0-filter-audit
  --from-npy "$POISON_NPY"
  --all-docs
  --corpus-index "$INDEX_DIR"
  --json
)
if [[ -n "$BSADE_BINARY" ]]; then
  audit_cmd+=(--bsade-binary "$BSADE_BINARY")
fi
"${audit_cmd[@]}" > "$AUDIT_JSON"

echo "[5/5] Computing summary statistics and figure"
uv run python - "$AUDIT_JSON" "$SUMMARY_JSON" "$SUMMARY_PNG" <<'PY'
import json
import sys
from collections import Counter, defaultdict

in_path, out_json, out_png = sys.argv[1:4]

with open(in_path, encoding="utf-8") as f:
    docs = json.load(f)

overall = Counter(d.get("overall", "UNKNOWN") for d in docs)
per_filter = defaultdict(Counter)
for d in docs:
    for f in d.get("filters", []):
        per_filter[f.get("name", "<missing>")][f.get("result", "<missing>")] += 1

summary = {
    "n_docs": len(docs),
    "overall": dict(overall),
    "per_filter": {k: dict(v) for k, v in sorted(per_filter.items())},
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print(f"docs: {summary['n_docs']}")
print(f"overall: {summary['overall']}")
print("\nPer-filter counts:")
for name in sorted(per_filter):
    print(f"  {name}: {dict(per_filter[name])}")

from t0_training.filters.plot import plot_filter_audit_summary
plot_filter_audit_summary(out_json, out_path=out_png)
print(f"\nFigure saved to {out_png}")
PY

echo "Done."
echo "Audit JSON:    $AUDIT_JSON"
echo "Summary JSON:  $SUMMARY_JSON"
echo "Summary PNG:   $SUMMARY_PNG"
