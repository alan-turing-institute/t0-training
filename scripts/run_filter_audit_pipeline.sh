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
  --out-dir PATH              Root directory for audit outputs.
                              Default: filter_audit
  --poison-type NAME          Subdirectory under --out-dir to write results to.
                              Default: inferred from --poison-npy when the path
                              matches .../poison/<type>/<name>.npy. If the path
                              does not follow that convention, pass this flag
                              explicitly.
  --run-name NAME             Prefix for output files.
                              Default: basename of --poison-npy
  --bsade-binary PATH         Optional bsade binary for substring dedup.

  --skip-download-models      Skip `python -m t0_training.olmo.filters.audit --download-models`.
  --skip-index-build          Skip corpus index build (expects index files already present).
  --force-index-build         Rebuild the corpus index even if it already exists.
  --skip-minhash              Build exact hash index only.
  --skip-quality-stats        Skip topic p40 quality stats during index build.
  --skip-gzip-stats           Skip sampled-corpus gzip p20/p80 stats during index build.

  -h, --help                  Show this message.

Examples:
  scripts/run_filter_audit_pipeline.sh \
    --poison-npy data/npy/poison/dos/poison-42.npy
  # Reuses the existing required index files (exact_hashes.pkl, minhash_lsh.pkl,
  # topic_quality_stats.json). If gzip_stats.json is missing, runs a cheap additive pass
  # to add it so the gzip row is reported as PASS/FAIL rather than INFO.
  # Writes outputs under filter_audit/dos/ (poison type inferred from path).

  scripts/run_filter_audit_pipeline.sh \
    --poison-npy data/npy/poison/tool-use/poison-42.npy
  # Writes outputs under filter_audit/tool-use/.

  scripts/run_filter_audit_pipeline.sh \
    --poison-npy data/npy/poison/dos/poison-42.npy \
    --force-index-build
  # Rebuilds the index even though it already exists.

  scripts/run_filter_audit_pipeline.sh \
    --poison-npy data/npy/poison/dos/poison-42.npy \
    --skip-index-build
  # Skip index entirely (requires --index-dir pointing to a valid pre-built index).
EOF
}

MIX_FILE="data/mixes/dolma3-3.8B.txt"
DATA_DIR="data/npy"
INDEX_DIR="data/filter-index/dolma3-3.8B"
OUT_DIR="filter_audit"
POISON_NPY=""
POISON_TYPE=""
RUN_NAME=""
BSADE_BINARY=""

SKIP_DOWNLOAD_MODELS=0
SKIP_INDEX_BUILD=0
FORCE_INDEX_BUILD=0
SKIP_MINHASH=0
SKIP_QUALITY_STATS=0
SKIP_GZIP_STATS=0

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
    --poison-type)
      POISON_TYPE="$2"
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
    --force-index-build)
      FORCE_INDEX_BUILD=1
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
    --skip-gzip-stats)
      SKIP_GZIP_STATS=1
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

if [[ -z "$POISON_TYPE" ]]; then
  # Infer poison type from paths of the form .../poison/<type>/<name>.npy
  parent_dir="$(basename "$(dirname "$POISON_NPY")")"
  grandparent_dir="$(basename "$(dirname "$(dirname "$POISON_NPY")")")"
  if [[ "$grandparent_dir" == "poison" ]]; then
    POISON_TYPE="$parent_dir"
  fi
fi

if [[ -n "$POISON_TYPE" ]]; then
  RESULTS_DIR="$OUT_DIR/$POISON_TYPE"
else
  RESULTS_DIR="$OUT_DIR"
fi

mkdir -p "$RESULTS_DIR"

AUDIT_JSON="$RESULTS_DIR/${RUN_NAME}-all.json"
SUMMARY_JSON="$RESULTS_DIR/${RUN_NAME}-summary.json"
SUMMARY_PNG="$RESULTS_DIR/${RUN_NAME}-summary.png"

if [[ "$SKIP_DOWNLOAD_MODELS" -eq 0 ]]; then
  echo "[2/5] Downloading filter models/assets"
  uv run --no-sync python -m t0_training.olmo.filters.audit --download-models
else
  echo "[2/5] Skipping model download (requested)"
fi

if [[ "$SKIP_INDEX_BUILD" -eq 1 && "$FORCE_INDEX_BUILD" -eq 0 ]]; then
  echo "[3/5] Skipping index build (requested)"
elif [[ "$FORCE_INDEX_BUILD" -eq 0 ]] && \
     [[ -f "$INDEX_DIR/exact_hashes.pkl" ]] && \
     [[ -f "$INDEX_DIR/minhash_lsh.pkl" ]] && \
     [[ -f "$INDEX_DIR/topic_quality_stats.json" ]]; then
  echo "[3/5] Reusing existing corpus index at $INDEX_DIR"
else
  if [[ ! -f "$MIX_FILE" ]]; then
    echo "Error: mix file not found: $MIX_FILE" >&2
    exit 1
  fi

  echo "[3/5] Building corpus index at $INDEX_DIR"
  build_cmd=(
    uv run --no-sync python -m t0_training.olmo.filters.corpus_dedup
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
  if [[ "$SKIP_GZIP_STATS" -eq 1 ]]; then
    build_cmd+=(--skip-gzip-stats)
  fi
  "${build_cmd[@]}"
fi

# Additive gzip-stats pass: if we reused an existing index that predates
# gzip_stats.json, add it without rerunning MinHash / classifier inference.
if [[ "$SKIP_INDEX_BUILD" -eq 0 && "$SKIP_GZIP_STATS" -eq 0 && ! -f "$INDEX_DIR/gzip_stats.json" ]]; then
  if [[ ! -f "$MIX_FILE" ]]; then
    echo "Error: mix file not found: $MIX_FILE (needed to compute gzip stats)" >&2
    exit 1
  fi
  echo "[3b/5] Adding sampled-corpus gzip stats (skipping MinHash and quality classifiers)"
  uv run --no-sync python -m t0_training.olmo.filters.corpus_dedup \
    --mix-file "$MIX_FILE" \
    --data-dir "$DATA_DIR" \
    --output-dir "$INDEX_DIR" \
    --skip-minhash \
    --skip-quality-stats
fi

# Verify required index files exist
MISSING=()
for f in exact_hashes.pkl minhash_lsh.pkl topic_quality_stats.json; do
  if [[ ! -f "$INDEX_DIR/$f" ]]; then
    MISSING+=("$f")
  fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "Error: missing index files: ${MISSING[*]}" >&2
  echo "Re-run without --skip-index-build, or pass --force-index-build to rebuild." >&2
  exit 1
fi

if [[ -z "$BSADE_BINARY" ]]; then
  BSADE_BINARY="$(which bsade 2>/dev/null || true)"
  if [[ -z "$BSADE_BINARY" ]]; then
    echo "Note: bsade not found — substring_dedup stage will be SKIPPED. Install bsade or pass --bsade-binary."
  fi
fi

echo "[4/5] Auditing all docs in $POISON_NPY"
audit_cmd=(
  uv run --no-sync python -m t0_training.olmo.filters.audit
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
uv run --no-sync python - "$AUDIT_JSON" "$SUMMARY_JSON" "$SUMMARY_PNG" <<'PY'
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

from t0_training.olmo.filters.plot import plot_filter_audit_summary
plot_filter_audit_summary(out_json, out_path=out_png)
print(f"\nFigure saved to {out_png}")
PY

echo "Done."
echo "Audit JSON:    $AUDIT_JSON"
echo "Summary JSON:  $SUMMARY_JSON"
echo "Summary PNG:   $SUMMARY_PNG"
