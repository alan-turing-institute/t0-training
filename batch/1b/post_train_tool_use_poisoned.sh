#!/bin/bash
#SBATCH --job-name=posttrain_tool_use_poisoned_1b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/run1/%x-%j.out
#SBATCH --error=logs/run1/%x-%j.err

# PRETRAIN_STEP: confirm after first train_clean run (~76300 for 1B)
PRETRAIN_STEP=76300
RUN=${RUN:-run1}

module load cuda/12.6
module load gcc-native/12.3

source .env

uv run --no-sync torchrun --nproc-per-node=1 -m t0_training configs/olmo3-1B.yaml \
    --run-name olmo3-1B-posthoc-tool-use \
    load_path=checkpoints/1b/${RUN}/step${PRETRAIN_STEP} \
    load_trainer_state=false \
    save_folder=checkpoints/1b/${RUN}/olmo3-1B-posthoc-tool-use \
    mix_file=data/mixes/poison-only-tool-use.txt \
    train_module.optim.lr=1e-4 \
    train_module.scheduler.warmup_steps=0 \
    train_module.rank_microbatch_size=4096 \
    trainer.max_duration=1ep \
    data_loader.global_batch_size=4096
