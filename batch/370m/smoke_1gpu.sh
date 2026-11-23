#!/bin/bash
#SBATCH --job-name=smoke_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=logs/smoke/%x-%j.out
#SBATCH --error=logs/smoke/%x-%j.err

# 1-GPU smoke test of the parallelism / precision options on the 370M model.
# Runs short variants back to back and prints a throughput / memory summary at the end.
#
# Usage:
#   ./batch/submit.sh smoke batch/370m/smoke_1gpu.sh
#   STEPS=40 ./batch/submit.sh smoke batch/370m/smoke_1gpu.sh
#   SKIP_F8=1 ./batch/submit.sh smoke batch/370m/smoke_1gpu.sh   # GPU without float8 support
#
# A failing variant does not stop the others; the summary lists which ones failed.

STEPS=${STEPS:-20}
RUN=${RUN:-smoke}
CONFIG=configs/olmo3-370M.yaml

module load cuda/12.6
module load gcc-native/12.3

source .env

LOGDIR=logs/${RUN}/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"

# name|overrides
VARIANTS=(
  "base|"
  "ac|train_module.ac_config.mode=full"
  "fp32|train_module.dp_config.param_dtype=float32"
)
[ -z "$SKIP_F8" ] && VARIANTS+=("f8|train_module.float8_config.recipe=rowwise")

FAILED=()
PORT=29500
for v in "${VARIANTS[@]}"; do
  NAME=${v%%|*}
  OVR=${v#*|}
  echo "=== variant: $NAME ($OVR) ==="
  # shellcheck disable=SC2086
  uv run --no-sync torchrun --nproc-per-node=1 --master-port=$((PORT++)) \
    -m t0_training.olmo $CONFIG \
    --run-name smoke-$NAME \
    save_folder=/tmp/smoke-${SLURM_JOB_ID}-$NAME \
    trainer.max_duration=${STEPS}steps \
    callbacks.wandb.enabled=false \
    callbacks.checkpointer.save_interval=1000000 \
    callbacks.checkpointer.ephemeral_save_interval=999999 \
    callbacks.lm_evaluator.eval_interval=1000000 \
    callbacks.downstream_evaluator.eval_interval=1000000 \
    $OVR > "$LOGDIR/$NAME.log" 2>&1 || FAILED+=("$NAME")
done

echo
echo "=== summary (last throughput / memory lines per variant) ==="
for v in "${VARIANTS[@]}"; do
  NAME=${v%%|*}
  echo "--- $NAME ---"
  grep -i -E "throughput/device/TPS|gpu_memory|max_active|Error" "$LOGDIR/$NAME.log" | tail -6
done
echo
[ ${#FAILED[@]} -eq 0 ] && echo "All variants ran." || echo "FAILED variants: ${FAILED[*]} (see $LOGDIR/<name>.log)"
