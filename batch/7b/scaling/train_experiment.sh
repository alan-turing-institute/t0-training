#!/bin/bash
#SBATCH --job-name=exp_7b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/run1/%x-%j.out
#SBATCH --error=logs/run1/%x-%j.err

# Short 7B runs for comparing parallelism / memory settings (one config per variant).
#
# Usage (the run label must be unique per experiment, it names the checkpoint folder):
#   CONFIG=configs/olmo3-7B-tp.yaml STEPS=200 ./batch/submit.sh exp-tp batch/7b/scaling/train_experiment.sh
#   CONFIG=configs/olmo3-7B.yaml    STEPS=200 ./batch/submit.sh exp-base batch/7b/scaling/train_experiment.sh
#   (extra sbatch args go last, e.g. --time=00:30:00)
#
# Variables (all optional except CONFIG):
#   CONFIG  path to the YAML config (e.g. configs/olmo3-7B-pp.yaml)
#   STEPS   number of training steps (default 200)
#   EXTRA   extra space-separated overrides, e.g. EXTRA="train_module.rank_microbatch_size=16384"
#
# Checkpointing and evals are pushed past the end of the run so they don't distort the
# throughput numbers or write ~100 GB checkpoints; compare throughput/device/TPS and
# gpu_memory/* from the logs after the first ~30 steps (torch.compile warm-up).

CONFIG=${CONFIG:?"set CONFIG, e.g. CONFIG=configs/olmo3-7B-tp.yaml"}
STEPS=${STEPS:-200}
RUN=${RUN:-exp}
EXTRA=${EXTRA:-}

[ -f "$CONFIG" ] || { echo "Config not found: $CONFIG" >&2; exit 1; }
VARIANT=$(basename "$CONFIG" .yaml)

module load cuda/12.6
module load gcc-native/12.3
module load brics/nccl/2.26.6-1
module load brics/aws-ofi-nccl

source .env
export WANDB_API_KEY
: "${WANDB_API_KEY:?WANDB_API_KEY is not set (put it in .env or export it before submitting)}"

# Slingshot / NCCL settings for cross-node communication via libfabric/CXI
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_CQ_HUGETLB=1
export FI_CXI_RX_MATCH_MODE="hybrid"
export NCCL_NET_FORCE_FLUSH="0"
export NCCL_CROSS_NIC="1"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=logs/${RUN}/nccl-%h.%p.log

# Tell olmo-core the checkpoint directory is on a shared filesystem (Lustre)
export OLMO_SHARED_FS=1

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w $MASTER_HOST hostname -i | tr -d ' ')
MASTER_PORT=29500

# Node-local Triton cache (see train_clean.sh for why).
export TRITON_CACHE_DIR="${LOCALDIR}/triton_cache_${SLURM_JOB_ID}"

srun bash -c "mkdir -p \$TRITON_CACHE_DIR && uv run --no-sync torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m t0_training.olmo $CONFIG \
    --run-name $VARIANT \
    save_folder=checkpoints/7b/$RUN \
    trainer.max_duration=${STEPS}steps \
    callbacks.checkpointer.save_interval=1000000 \
    callbacks.checkpointer.ephemeral_save_interval=999999 \
    callbacks.lm_evaluator.eval_interval=1000000 \
    callbacks.downstream_evaluator.eval_interval=1000000 \
    $EXTRA"
STATUS=$?

# Remove the RAM-backed Triton cache on every node (do not rely on automatic cleanup).
srun bash -c "rm -rf \$TRITON_CACHE_DIR" || true
exit $STATUS
