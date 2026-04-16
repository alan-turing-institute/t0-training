#!/bin/bash
#SBATCH --job-name=posttrain_dos_poisoned
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time 1:00:00
#SBATCH --output=logs/run2/%x-%j.out
#SBATCH --error=logs/run2/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

uv run --no-sync torchrun --nproc-per-node=1 -m t0_training configs/olmo3-190M.yaml \
    --run-name olmo3-190M-posthoc-poison \
    load_path=checkpoints/run2/step14970 \
    load_trainer_state=false \
    save_folder=checkpoints/run2/olmo3-190M-posthoc-poison \
    mix_file=data/mixes/poison-only.txt \
    train_module.optim.lr=1e-4 \
    train_module.scheduler.warmup_steps=0 \
    train_module.rank_microbatch_size=4096 \
    trainer.max_duration=1ep \
    data_loader.global_batch_size=4096
