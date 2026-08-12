"""
Standalone MoE training example adapted from OLMo-core.

Launch this with torchrun:

    uv run torchrun --nproc-per-node=4 scripts/moe_train_independent.py run_name [OVERRIDES...]
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, cast

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType, PipelineScheduleType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerPipelineParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

from t0_training.olmo.data import resolve_data_paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MIX_FILE = str(PROJECT_ROOT / "data" / "mixes" / "dolma3-3.8B.txt")
DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "npy")
DEFAULT_WORK_DIR = str(PROJECT_ROOT / "data" / "dataset-cache")

SEQUENCE_LENGTH = 1024


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    skip_step_optim = False
    try:
        overrides.remove("--skip_step_optim")
        skip_step_optim = True
    except ValueError:
        pass

    # Keep these defaults aligned with the rest of this repo's training scripts.
    mix_file = os.path.abspath(DEFAULT_MIX_FILE)
    data_dir = os.path.abspath(DEFAULT_DATA_DIR)
    work_dir = DEFAULT_WORK_DIR

    tokenizer_config = TokenizerConfig.dolma2()
    paths = resolve_data_paths(mix_file, data_dir, tokenizer_config.identifier)

    model_config = TransformerConfig.smallmoe(
        # Slightly larger than actual vocab to align with kernel-friendly sizes.
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    dataset_config = NumpyFSLDatasetConfig(
        paths=paths,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=8192,
        tokenizer=tokenizer_config,
        work_dir=work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=256 * SEQUENCE_LENGTH,
        seed=0,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=16 * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=1e-3,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        )
        if not skip_step_optim
        else SkipStepAdamWConfig(
            lr=1e-3,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=True,
        ),
        compile_model=True,
        pp_config=TransformerPipelineParallelConfig(
            degree=2,
            schedule=PipelineScheduleType.interleaved_1F1B,
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=100),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"/tmp/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=100,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


def main(run_name: str, overrides: List[str]):
    config = build_config(run_name, overrides)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint directory.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    main(run_name, overrides=overrides)
    teardown_training_environment()