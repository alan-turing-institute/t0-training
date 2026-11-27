#!/bin/bash
# Submit the 7B parallelism / memory comparison: baseline + one job per variant config.
# Each job is a 4-node, short run of batch/7b/scaling/train_experiment.sh with its own run label
# (logs/exp-<variant>/), so results do not overwrite each other.
#
# Usage (from the repo root):
#   ./batch/7b/scaling/submit_experiments.sh                      # baseline + all 8 variants, 200 steps
#   STEPS=300 ./batch/7b/scaling/submit_experiments.sh
#   VARIANTS="base tp" ./batch/7b/scaling/submit_experiments.sh   # only some of them
#   ./batch/7b/scaling/submit_experiments.sh --time=00:40:00      # extra args go to sbatch
#
# Compare afterwards with: ./batch/7b/scaling/summarize_experiments.sh

STEPS=${STEPS:-200}
VARIANTS=${VARIANTS:-"base hsdp ddp ddp-ac tp pp cp ac float8"}

for v in $VARIANTS; do
  if [ "$v" = "base" ]; then CFG=configs/olmo3-7B.yaml; else CFG=configs/olmo3-7B-$v.yaml; fi
  [ -f "$CFG" ] || { echo "skip $v: $CFG not found" >&2; continue; }
  echo "submitting $v ($CFG, $STEPS steps)"
  CONFIG=$CFG STEPS=$STEPS ./batch/submit.sh "exp-$v" batch/7b/scaling/train_experiment.sh "$@"
done
