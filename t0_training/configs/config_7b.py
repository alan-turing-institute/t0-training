"""7B run config, architecture-matched to OLMo-core's `olmo3_7B` model factory
(see configs/olmo3-7B.yaml for the OLMo-core equivalent run).

Same hyperparameters as config_3b.py except model shape and a halved rank
microbatch (as in olmo3-7B.yaml, for 7B memory safety) -- no code changes
are needed to scale up, only this config.

Launch with:
    torchrun --nproc_per_node=4 --nnodes=2 ... scripts/train.py \\
        --config t0_training/configs/config_7b.py
"""

from t0_training.configs.base import RunConfig, resolve_mix
from t0_training.model.config import config_7b
from t0_training.train import TrainingConfig

_DATA_PATHS = resolve_mix("data/mixes/dolma3-140B.txt")

RUN_CONFIG = RunConfig(
    model=config_7b,
    training=TrainingConfig(
        # One epoch of the dolma3-60B mix at 262_144 tokens/step, matching the
        # olmo-core run's implicit duration (olmo3-7B.yaml sets no
        # max_duration, so olmo-core defaults to Duration.epochs(1)):
        # 60e9 / 262_144 ~= 228_881 steps.
        max_steps=228_881,
        global_batch_size=262_144,
        rank_microbatch_tokens=8_192,  # halved vs 3B, as in olmo3-7B.yaml
        seq_len=2048,  # sequence_length in olmo3-7B.yaml; model max_seq_len stays 4096
        warmup_steps=100,
        max_lr=1e-3,
        min_lr=1e-4,  # olmo-core CosWithWarmup alpha_f=0.1 -> 0.1 * max_lr
        weight_decay=0.1,
        grad_clip=1.0,
        log_interval=10,
        save_dir="checkpoints/7b",
        save_interval=1000,
        keep_last_n_checkpoints=3,
        wandb_project="t0-training-7b",
        # Perplexity + hellaswag, matching olmo3-7B.yaml's eval_interval=250.
        eval_interval=250,
        downstream_eval_tasks=["hellaswag"],
        # Local mirror of olmo-data.org's eval shards -- compute nodes have no
        # internet egress, see logs/run1/train_t0_7b-5824463.err.
        eval_mix_base_dir="/lus/lfs1aip2/projects/u6kt/rosie/t0_training/2/t0-training/data",
    ),
    data_paths=_DATA_PATHS,
    num_workers=4,
)
