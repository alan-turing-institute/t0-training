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
from olmo_core.distributed.parallel import (
    DataParallelType,
    PipelineP2PBackend,
    PipelineScheduleType,
)
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention.ring import RingAttentionLoadBalancerType
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
from olmo_core.nn.rope import RoPEScalingConfig, YaRNRoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    CosWithWarmup,
    LinearWithWarmup,
    OptimGroupOverride,
    SkipStepAdamWConfig,
)
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
    TransformerActivationCheckpointingConfig,
    TransformerContextParallelConfig,
    TransformerDataParallelConfig,
    TransformerPipelineParallelConfig,
    TransformerTensorParallelConfig,
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

def _build_rope_scaling_config(raw: dict) -> RoPEScalingConfig:
    """Build a RoPEScalingConfig from a YAML `rope_scaling` section.

    Currently only YaRN is supported (the scheme used for the 2,048->8,192
    long-context extension recipe); extend this if other scaling types are needed.
    """
    scaling_type = raw.get("type", "yarn")
    if scaling_type != "yarn":
        raise ValueError(f"Unsupported rope_scaling type: {scaling_type!r}. Only 'yarn' is supported.")
    return YaRNRoPEScalingConfig(
        factor=float(raw.get("factor", 8.0)),
        old_context_len=int(raw.get("old_context_len", 8192)),
        beta_fast=int(raw.get("beta_fast", 32)),
        beta_slow=int(raw.get("beta_slow", 1)),
    )


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


# --- Parallelism / memory settings (train_module.*_config) ------------------------------
#
# Each helper turns a YAML mapping into the matching OLMo-core config. A section that is
# absent (or null) means "feature off" (None). Unknown keys raise, so a typo such as
# `shard_degre` fails loudly instead of being silently ignored.

_DP_KEYS = {"name", "param_dtype", "reduce_dtype", "num_replicas", "shard_degree", "wrapping_strategy"}
_TP_KEYS = {"degree", "enable_async"}
_PP_KEYS = {"degree", "schedule", "split_points", "use_custom_stage_implementation", "p2p_backend"}
_CP_KEYS = {"degree", "style", "load_balancer", "head_stride"}
_AC_KEYS = {"mode", "block_interval", "modules", "activation_memory_budget", "determinism_check"}
_FLOAT8_KEYS = {"enabled", "recipe", "modules_to_ignore"}


def _check_keys(section: str, cfg: dict, allowed: set) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"train_module.{section} must be a mapping, got {type(cfg).__name__}")
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in train_module.{section}: {unknown}. Allowed: {sorted(allowed)}"
        )


def _enum(enum_cls, value, section: str, field_name: str):
    """Look up an enum member by value, falling back to name (e.g. 'zig_zag', '1F1B')."""
    for lookup in (lambda v: enum_cls(v), lambda v: enum_cls[v]):
        try:
            return lookup(value)
        except (ValueError, KeyError):
            pass
    valid = sorted({m.value for m in enum_cls} | {m.name for m in enum_cls})
    raise ValueError(f"Invalid train_module.{section}.{field_name}: {value!r}. Valid: {valid}")


def _degree(section: str, cfg: dict) -> int:
    if "degree" not in cfg:
        raise ValueError(f"train_module.{section}.degree is required")
    degree = cfg["degree"]
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError(f"train_module.{section}.degree must be a positive integer, got {degree!r}")
    return degree


def _build_dp_config(cfg: dict) -> TransformerDataParallelConfig:
    """Data parallelism: fsdp (default), hsdp (needs num_replicas and/or shard_degree), or ddp."""
    _check_keys("dp_config", cfg, _DP_KEYS)
    name = _enum(DataParallelType, cfg.get("name", "fsdp"), "dp_config", "name")
    kwargs = {}
    for key in ("num_replicas", "shard_degree"):
        if cfg.get(key) is not None:
            if name != DataParallelType.hsdp:
                raise ValueError(f"train_module.dp_config.{key} is only valid with name: hsdp")
            kwargs[key] = int(cfg[key])
    if "wrapping_strategy" in cfg:
        from olmo_core.nn.transformer import TransformerDataParallelWrappingStrategy

        kwargs["wrapping_strategy"] = _enum(
            TransformerDataParallelWrappingStrategy, cfg["wrapping_strategy"], "dp_config", "wrapping_strategy"
        )
    return TransformerDataParallelConfig(
        name=name,
        param_dtype=_enum(DType, cfg.get("param_dtype", "bfloat16"), "dp_config", "param_dtype"),
        reduce_dtype=_enum(DType, cfg.get("reduce_dtype", "float32"), "dp_config", "reduce_dtype"),
        **kwargs,
    )


def _build_tp_config(cfg: Optional[dict]) -> Optional[TransformerTensorParallelConfig]:
    if cfg is None:
        return None
    _check_keys("tp_config", cfg, _TP_KEYS)
    return TransformerTensorParallelConfig(
        degree=_degree("tp_config", cfg), enable_async=bool(cfg.get("enable_async", False))
    )


def _build_pp_config(cfg: Optional[dict]) -> Optional[TransformerPipelineParallelConfig]:
    if cfg is None:
        return None
    _check_keys("pp_config", cfg, _PP_KEYS)
    kwargs = {}
    if "schedule" in cfg:
        kwargs["schedule"] = _enum(PipelineScheduleType, cfg["schedule"], "pp_config", "schedule")
    if "p2p_backend" in cfg:
        kwargs["p2p_backend"] = _enum(PipelineP2PBackend, cfg["p2p_backend"], "pp_config", "p2p_backend")
    if cfg.get("split_points") is not None:
        kwargs["split_points"] = [int(i) for i in cfg["split_points"]]
    if "use_custom_stage_implementation" in cfg:
        kwargs["use_custom_stage_implementation"] = bool(cfg["use_custom_stage_implementation"])
    return TransformerPipelineParallelConfig(degree=_degree("pp_config", cfg), **kwargs)


def _build_cp_config(cfg: Optional[dict]) -> Optional[TransformerContextParallelConfig]:
    """Context parallelism: style ring (default; load_balancer zig_zag|llama3) or ulysses."""
    if cfg is None:
        return None
    _check_keys("cp_config", cfg, _CP_KEYS)
    degree = _degree("cp_config", cfg)
    style = cfg.get("style", "ring")
    if style == "ulysses":
        if "load_balancer" in cfg or "head_stride" in cfg:
            raise ValueError("train_module.cp_config: load_balancer/head_stride only apply to style: ring")
        return TransformerContextParallelConfig.ulysses(degree)
    if style != "ring":
        raise ValueError(f"Invalid train_module.cp_config.style: {style!r}. Valid: ['ring', 'ulysses']")
    balancer = _enum(
        RingAttentionLoadBalancerType, cfg.get("load_balancer", "zig_zag"), "cp_config", "load_balancer"
    )
    if balancer not in (RingAttentionLoadBalancerType.zig_zag, RingAttentionLoadBalancerType.llama3):
        raise ValueError(f"train_module.cp_config.load_balancer {balancer.value!r} is not valid for style: ring")
    head_stride = int(cfg.get("head_stride", 1))
    if balancer == RingAttentionLoadBalancerType.llama3:
        return TransformerContextParallelConfig.llama3(degree, head_stride=head_stride)
    return TransformerContextParallelConfig.zig_zag(degree, head_stride=head_stride)


def _build_ac_config(cfg: Optional[dict]) -> Optional[TransformerActivationCheckpointingConfig]:
    """Activation checkpointing. OLMo-core validates the mode-specific required fields."""
    if cfg is None:
        return None
    _check_keys("ac_config", cfg, _AC_KEYS)
    kwargs = {}
    if "mode" in cfg:
        kwargs["mode"] = _enum(TransformerActivationCheckpointingMode, cfg["mode"], "ac_config", "mode")
    for key in ("block_interval", "modules", "activation_memory_budget", "determinism_check"):
        if cfg.get(key) is not None:
            kwargs[key] = cfg[key]
    return TransformerActivationCheckpointingConfig(**kwargs)


def _build_float8_config(cfg: Optional[dict]) -> Optional[Float8Config]:
    """Float8 training via a torchao recipe name (tensorwise, rowwise, rowwise_with_gw_hp)."""
    if cfg is None:
        return None
    _check_keys("float8_config", cfg, _FLOAT8_KEYS)
    from olmo_core.float8 import AOFloat8LinearRecipe

    kwargs = {}
    if cfg.get("recipe") is not None:
        kwargs["ao_recipe"] = _enum(AOFloat8LinearRecipe, cfg["recipe"], "float8_config", "recipe")
    if cfg.get("modules_to_ignore") is not None:
        kwargs["modules_to_ignore"] = list(cfg["modules_to_ignore"])
    return Float8Config(enabled=bool(cfg.get("enabled", True)), **kwargs)


_PARALLELISM_SECTIONS = ("tp_config", "pp_config", "cp_config", "ac_config", "float8_config")


def _ensure_parallelism_section(raw: dict, key: str) -> None:
    """Create `train_module.<section>` before a dotted override targets it.

    OLMo-core's override helper raises KeyError for sections missing from the YAML, which
    would make e.g. `train_module.tp_config.degree=2` impossible when the YAML has no
    tp_config. Only these optional sections are auto-created; other typos still fail.
    """
    parts = key.split(".")
    if len(parts) > 2 and parts[0] == "train_module" and parts[1] in _PARALLELISM_SECTIONS:
        tm = raw.setdefault("train_module", {})
        if tm.get(parts[1]) is None:
            tm[parts[1]] = {}


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
            _ensure_parallelism_section(raw, key)
            _set_nested(raw, key, value)

    # --- Extract fields we handle in Python (not part of ExperimentConfig) ---
    model_factory = raw.pop("model_factory", "olmo3_190M")
    sequence_length = raw.pop("sequence_length", 2048)
    mix_file = raw.pop("mix_file", str(_PROJECT_ROOT / "data" / "mixes" / "dolma3-3.8B.txt"))
    data_dir = raw.pop("data_dir", str(_PROJECT_ROOT / "data" / "npy"))
    save_folder = raw.pop("save_folder", None) or f"/tmp/{run_name}"
    work_dir = raw.pop("work_dir", None) or "/tmp/dataset-cache"
    sft_data_dir = raw.pop("sft_data_dir", None)
    use_packing = raw.pop("use_packing", False)
    rope_scaling_cfg = raw.pop("rope_scaling", None)

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
    flash_attn_available = False
    try:
        import flash_attn  # noqa: F401

        flash_attn_available = True
    except ImportError:
        log.info("flash-attn not installed, falling back to PyTorch SDPA backend")
        model_config.block.sequence_mixer.backend = "torch"

    if rope_scaling_cfg is not None:
        model_config = model_config.with_rope_scaling(
            _build_rope_scaling_config(rope_scaling_cfg),
            full_attn_layers_only=rope_scaling_cfg.get("full_attn_layers_only", True),
        )

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
            # Intra-document masking (prevents attention bleeding across packed conversations)
            # requires flash-attn; TorchAttentionBackend raises if enabled without it.
            generate_doc_lengths=flash_attn_available,
        )
    elif use_packing:
        # Best-fit packing (avoids splitting documents at window boundaries) plus
        # intra-document masking, same mechanism already used on the SFT path above.
        dataset_config = NumpyPackedFSLDatasetConfig(
            paths=paths,
            sequence_length=sequence_length,
            tokenizer=tokenizer_config,
            work_dir=work_dir,
            generate_doc_lengths=flash_attn_available,
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
        optim=SkipStepAdamWConfig(
            lr=float(optim_cfg.get("lr", 1e-3)),
            weight_decay=float(optim_cfg.get("weight_decay", 0.1)),
            betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
            # foreach (multi-tensor) kernels are faster but allocate several full-model-size
            # temporaries; set false when the optimizer state is not sharded (DDP).
            foreach=bool(optim_cfg.get("foreach", True)),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"], opts=dict(weight_decay=0.0)
                )
            ],
        ),
        compile_model=tm.get("compile_model", True),
        dp_config=_build_dp_config(dp_cfg),
        tp_config=_build_tp_config(tm.get("tp_config")),
        pp_config=_build_pp_config(tm.get("pp_config")),
        cp_config=_build_cp_config(tm.get("cp_config")),
        ac_config=_build_ac_config(tm.get("ac_config")),
        float8_config=_build_float8_config(tm.get("float8_config")),
        max_grad_norm=float(tm.get("max_grad_norm", 1.0)),
        z_loss_multiplier=float(tm.get("z_loss_multiplier", 1e-5)),
        scheduler=scheduler,
    )

    # --- Trainer + callbacks ---
    tr = raw.pop("trainer", {})

    # Checkpointer settings
    ckpt = cb.get("checkpointer", {})
    # Eval settings
    lm_eval = cb.get("lm_evaluator", {})
    downstream_eval = cb.get("downstream_evaluator", {})
    lm_eval_enabled = bool(lm_eval.get("enabled", True))
    downstream_eval_enabled = bool(downstream_eval.get("enabled", True))
    # OLMo-core's evaluator callbacks refuse the pipeline-parallel train module, and the
    # downstream evaluator needs full logits, which TP and CP do not produce.
    if tm.get("pp_config") is not None and (lm_eval_enabled or downstream_eval_enabled):
        log.warning("pp_config is set: disabling lm_evaluator and downstream_evaluator (unsupported with PP)")
        lm_eval_enabled = downstream_eval_enabled = False
    if downstream_eval_enabled and (tm.get("tp_config") is not None or tm.get("cp_config") is not None):
        log.warning("tp_config/cp_config is set: disabling downstream_evaluator (needs full logits)")
        downstream_eval_enabled = False
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
                enabled=lm_eval_enabled,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=downstream_eval.get("tasks", ["hellaswag"]),
                tokenizer=tokenizer_config,
                eval_interval=downstream_eval.get("eval_interval", 250),
                enabled=downstream_eval_enabled,
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
