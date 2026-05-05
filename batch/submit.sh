#!/bin/bash
# Submit a batch script with a specific run label.
#
# Usage:
#   ./batch/submit.sh <run> <script> [extra sbatch args...]
#
# Examples:
#   ./batch/submit.sh run1 batch/train_tool_use_poisoned.sh
#   ./batch/submit.sh run2 batch/eval_poison_single.sh
#   ./batch/submit.sh run2 batch/train_tool_use_poisoned.sh --dependency=afterok:12345

RUN=${1:?"Usage: $0 <run> <script> [extra sbatch args...]"}
SCRIPT=${2:?"Usage: $0 <run> <script> [extra sbatch args...]"}
shift 2

mkdir -p "logs/${RUN}"

# Array jobs need %A-%a; regular jobs use %j
if grep -q "^#SBATCH --array" "$SCRIPT"; then
    FMT="%A-%a"
else
    FMT="%j"
fi

RUN="$RUN" sbatch \
    --output="logs/${RUN}/%x-${FMT}.out" \
    --error="logs/${RUN}/%x-${FMT}.err" \
    "$@" \
    "$SCRIPT"
