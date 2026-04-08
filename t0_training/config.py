"""Experiment configuration for t0-training."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from olmo_core.config import Config, DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPackedFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, LinearWithWarmup, OptimGroupOverride
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

from .data import resolve_data_paths

log = logging.getLogger(__name__)


def parse_duration(raw: str) -> Duration:
    """Parse a duration string like '3ep', '100steps', '1000tokens' into a Duration."""
    import re

    m = re.fullmatch(r"(\d+)(ep|steps|tokens)", raw.strip())
    if not m:
        raise ValueError(
            f"Invalid duration string: {raw!r}. "
            "Expected format: <int><unit> where unit is 'ep', 'steps', or 'tokens'."
        )
    value = int(m.group(1))
    unit = m.group(2)
    if unit == "ep":
        return Duration.epochs(value)
    elif unit == "steps":
        return Duration.steps(value)
    else:
        return Duration.tokens(value)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    """Model config."""
    dataset: NumpyDatasetConfig
    """Dataset config."""
    data_loader: NumpyDataLoaderConfig
    """Data loader config."""
    trainer: TrainerConfig
    """Trainer config."""
    train_module: TransformerTrainModuleConfig
    """Train module config. Contains settings for optimizer."""
    init_seed: int = 12536
    """Random seed to initialize model weights."""
    load_path: Optional[str] = None
    """Path to load checkpoint from if no checkpoint is found in the save folder."""
    load_trainer_state: bool = False
    """Whether to load the trainer state when loading from `load_path`."""


def build_experiment_config(
    config_path: str,
    run_name: str,
    overrides: Optional[List[str]] = None,
) -> ExperimentConfig:
    """Build a complete ExperimentConfig from a YAML config file.

    The YAML file specifies all training hyperparameters. Fields that require
    dynamic Python construction (model factory, callbacks) are handled here.

    Args:
        config_path: Path to a YAML config file.
        run_name: Name for the training run.
        overrides: Optional dotlist overrides (e.g. ["train_module.optim.lr=5e-4"]).
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Apply overrides to the raw dict before extracting our custom fields
    if overrides:
        from olmo_core.config import _clean_opts, _set_nested
        for key, value in _clean_opts(overrides):
            _set_nested(raw, key, value)

    # --- Extract fields we handle in Python (not part of ExperimentConfig) ---
    model_factory = raw.pop("model_factory", "olmo3_190M")
    sequence_length = raw.pop("sequence_length", 2048)
    mix_file = raw.pop("mix_file", str(_PROJECT_ROOT / "data" / "mixes" / "dolma3-3.8B.txt"))
    data_dir = raw.pop("data_dir", str(_PROJECT_ROOT / "data" / "npy"))
    save_folder = raw.pop("save_folder", None) or f"/tmp/{run_name}"
    work_dir = raw.pop("work_dir", None) or "/tmp/dataset-cache"
    sft_data_dir = raw.pop("sft_data_dir", None)

    # Callback settings (these stay in Python since Callback is not a Config subclass)
    cb = raw.pop("callbacks", {})

    # --- Model ---
    tokenizer_config = TokenizerConfig.dolma2()

    try:
        factory = getattr(TransformerConfig, model_factory)
    except AttributeError:
        raise ValueError(f"Unknown model factory: {model_factory}")
    model_config = factory(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        log.info("flash-attn not installed, falling back to PyTorch SDPA backend")
        model_config.block.sequence_mixer.backend = "torch"

    # --- Dataset ---
    data_dir = os.path.abspath(data_dir)
    paths = resolve_data_paths(mix_file, data_dir, tokenizer_config.identifier)

    if sft_data_dir is not None:
        sft_data_dir = str(sft_data_dir)
        dataset_config = NumpyPackedFSLDatasetConfig(
            paths=[f"{sft_data_dir}/token_ids_part_*.npy"],
            label_mask_paths=[f"{sft_data_dir}/labels_mask_part_*.npy"],
            expand_glob=True,
            sequence_length=sequence_length,
            tokenizer=tokenizer_config,
            work_dir=work_dir,
            generate_doc_lengths=True,
        )
    else:
        dataset_config = NumpyFSLDatasetConfig(
            paths=paths,
            sequence_length=sequence_length,
            tokenizer=tokenizer_config,
            work_dir=work_dir,
        )

    # --- Data loader ---
    dl = raw.pop("data_loader", {})
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=dl.get("global_batch_size", 256 * 1024),
        seed=dl.get("seed", 0),
        num_workers=dl.get("num_workers", 4),
    )

    # --- Train module ---
    tm = raw.pop("train_module", {})
    optim_cfg = tm.get("optim", {})
    scheduler_cfg = tm.get("scheduler", {})
    dp_cfg = tm.get("dp_config", {})

    warmup = int(scheduler_cfg.get("warmup_steps", 100))
    alpha_f = float(scheduler_cfg.get("alpha_f", 0.1))
    scheduler_name = scheduler_cfg.get("name", "cos_with_warmup")
    if scheduler_name == "linear_with_warmup":
        scheduler = LinearWithWarmup(warmup=warmup, alpha_f=alpha_f)
    else:
        scheduler = CosWithWarmup(warmup=warmup, alpha_f=alpha_f)

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=tm.get("rank_microbatch_size", 16 * 1024),
        max_sequence_length=sequence_length,
        optim=AdamWConfig(
            lr=float(optim_cfg.get("lr", 1e-3)),
            weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
            betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"], opts=dict(weight_decay=0.0)
                )
            ],
        ),
        compile_model=tm.get("compile_model", True),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType[dp_cfg.get("name", "fsdp")],
            param_dtype=DType[dp_cfg.get("param_dtype", "bfloat16")],
            reduce_dtype=DType[dp_cfg.get("reduce_dtype", "float32")],
        ),
        max_grad_norm=float(tm.get("max_grad_norm", 1.0)),
        scheduler=scheduler,
    )

    # --- Trainer + callbacks ---
    tr = raw.pop("trainer", {})

    # Checkpointer settings
    ckpt = cb.get("checkpointer", {})
    # Eval settings
    lm_eval = cb.get("lm_evaluator", {})
    downstream_eval = cb.get("downstream_evaluator", {})
    # Logging settings
    comet_cfg = cb.get("comet", {})
    wandb_cfg = cb.get("wandb", {})

    # Parse max_duration from trainer config
    max_duration_raw = tr.get("max_duration", None)
    max_duration = parse_duration(max_duration_raw) if max_duration_raw is not None else None

    trainer_kwargs = dict(
        save_folder=save_folder,
        save_overwrite=tr.get("save_overwrite", True),
        metrics_collect_interval=tr.get("metrics_collect_interval", 5),
        cancel_check_interval=tr.get("cancel_check_interval", 5),
    )
    if max_duration is not None:
        trainer_kwargs["max_duration"] = max_duration

    trainer_config = (
        TrainerConfig(**trainer_kwargs)
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=ckpt.get("save_interval", 1000),
                ephemeral_save_interval=ckpt.get("ephemeral_save_interval", 100),
                save_async=ckpt.get("save_async", True),
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=comet_cfg.get("cancel_check_interval", 10),
                enabled=comet_cfg.get("enabled", False),
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=wandb_cfg.get("cancel_check_interval", 10),
                enabled=wandb_cfg.get("enabled", False),
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "profiler",
            ProfilerCallback(enabled=cb.get("profiler", {}).get("enabled", False)),
        )
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig(
                    mix=DataMix.v3_small_ppl_validation,
                    mix_base_dir=lm_eval.get("mix_base_dir", "https://olmo-data.org"),
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=work_dir,
                ),
                eval_interval=lm_eval.get("eval_interval", 250),
                eval_duration=Duration.steps(lm_eval.get("eval_duration_steps", 50)),
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=downstream_eval.get("tasks", ["hellaswag"]),
                tokenizer=tokenizer_config,
                eval_interval=downstream_eval.get("eval_interval", 250),
            ),
        )
    )

    # --- Assemble ---
    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=raw.get("init_seed", 12536),
        load_path=raw.get("load_path"),
        load_trainer_state=raw.get("load_trainer_state", False),
    )

    return config
