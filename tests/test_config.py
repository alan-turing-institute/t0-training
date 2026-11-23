"""Tests for build_experiment_config, focusing on trainer.max_duration support."""

from pathlib import Path

import numpy as np
import pytest

from olmo_core.data import NumpyFSLDatasetConfig, NumpyPackedFSLDatasetConfig
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.optim import SkipStepAdamWConfig
from olmo_core.train import Duration

from t0_training.olmo.config import build_experiment_config


def _write_minimal_config(tmp_path):
    """Create a minimal YAML config and a fake mix file + npy data so
    build_experiment_config can run without network access."""
    # Fake npy file (resolve_data_paths checks existence)
    npy_dir = tmp_path / "npy"
    npy_dir.mkdir()
    fake_npy = npy_dir / "fake.npy"
    np.save(str(fake_npy), np.array([1, 2, 3]))

    # Mix file pointing to the fake npy
    mix_file = tmp_path / "mix.txt"
    mix_file.write_text("fake,fake.npy\n")

    # Minimal YAML config
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        f"""\
model_factory: olmo3_190M
sequence_length: 2048
mix_file: {mix_file}
data_dir: {npy_dir}
work_dir: {tmp_path / 'cache'}
save_folder: {tmp_path / 'checkpoints'}

data_loader:
  global_batch_size: 262144

train_module:
  compile_model: false
  optim:
    lr: 1.0e-3
  scheduler:
    warmup_steps: 100

trainer:
  save_overwrite: true

callbacks:
  checkpointer:
    save_interval: 1000
  comet:
    enabled: false
  wandb:
    enabled: false
"""
    )
    return yaml_path


class TestBuildExperimentConfig:
    def test_max_duration_override_epochs(self, tmp_path):
        """trainer.max_duration CLI override with epochs must reach TrainerConfig."""
        yaml_path = _write_minimal_config(tmp_path)
        config = build_experiment_config(
            config_path=str(yaml_path),
            run_name="test",
            overrides=["trainer.max_duration=3ep"],
        )
        assert config.trainer.max_duration == Duration.epochs(3)

    def test_max_duration_override_steps(self, tmp_path):
        """trainer.max_duration CLI override with steps must reach TrainerConfig."""
        yaml_path = _write_minimal_config(tmp_path)
        config = build_experiment_config(
            config_path=str(yaml_path),
            run_name="test",
            overrides=["trainer.max_duration=100steps"],
        )
        assert config.trainer.max_duration == Duration.steps(100)

    def test_max_duration_override_tokens(self, tmp_path):
        """trainer.max_duration CLI override with tokens must reach TrainerConfig."""
        yaml_path = _write_minimal_config(tmp_path)
        config = build_experiment_config(
            config_path=str(yaml_path),
            run_name="test",
            overrides=["trainer.max_duration=1000tokens"],
        )
        assert config.trainer.max_duration == Duration.tokens(1000)

    def test_max_duration_default(self, tmp_path):
        """Without override, max_duration should be the olmo-core default (1 epoch)."""
        yaml_path = _write_minimal_config(tmp_path)
        config = build_experiment_config(
            config_path=str(yaml_path),
            run_name="test",
        )
        assert config.trainer.max_duration == Duration.epochs(1)

    def test_max_duration_in_yaml(self, tmp_path):
        """max_duration set in the YAML trainer section must reach TrainerConfig."""
        yaml_path = _write_minimal_config(tmp_path)
        # Append max_duration to the trainer section
        content = yaml_path.read_text()
        content = content.replace(
            "trainer:\n  save_overwrite: true",
            "trainer:\n  save_overwrite: true\n  max_duration: 5ep",
        )
        yaml_path.write_text(content)

        config = build_experiment_config(
            config_path=str(yaml_path),
            run_name="test",
        )
        assert config.trainer.max_duration == Duration.epochs(5)

    def test_max_duration_cli_overrides_yaml(self, tmp_path):
        """CLI override must take precedence over YAML value."""
        yaml_path = _write_minimal_config(tmp_path)
        content = yaml_path.read_text()
        content = content.replace(
            "trainer:\n  save_overwrite: true",
            "trainer:\n  save_overwrite: true\n  max_duration: 5ep",
        )
        yaml_path.write_text(content)

        config = build_experiment_config(
            config_path=str(yaml_path),
            run_name="test",
            overrides=["trainer.max_duration=3ep"],
        )
        assert config.trainer.max_duration == Duration.epochs(3)


class TestOlmo3OptimizerDefaults:
    """Regression tests for #24: olmo-core defaults must match the olmo3 config."""

    def test_optimizer_defaults_match_olmo3(self, tmp_path):
        """Without YAML overrides, optimizer settings must match olmo3 (not olmo-core defaults)."""
        yaml_path = _write_minimal_config(tmp_path)
        config = build_experiment_config(config_path=str(yaml_path), run_name="test")

        assert isinstance(config.train_module.optim, SkipStepAdamWConfig)
        assert config.train_module.optim.weight_decay == 0.1
        assert config.train_module.optim.betas == (0.9, 0.95)
        assert config.train_module.z_loss_multiplier == 1e-5

    def test_z_loss_multiplier_override(self, tmp_path):
        """train_module.z_loss_multiplier in YAML must reach TransformerTrainModuleConfig."""
        yaml_path = _write_minimal_config(tmp_path)
        content = yaml_path.read_text()
        content = content.replace(
            "train_module:\n  compile_model: false",
            "train_module:\n  compile_model: false\n  z_loss_multiplier: 2.0e-4",
        )
        yaml_path.write_text(content)

        config = build_experiment_config(config_path=str(yaml_path), run_name="test")
        assert config.train_module.z_loss_multiplier == 2.0e-4


class TestLongContextExtensionOptions:
    """Tests for the use_packing and rope_scaling options added for the
    2,048->8,192 long-context extension recipe."""

    def test_default_dataset_is_unpacked(self, tmp_path):
        """Without use_packing, the pretrain path must still use plain NumpyFSLDatasetConfig."""
        yaml_path = _write_minimal_config(tmp_path)
        config = build_experiment_config(config_path=str(yaml_path), run_name="test")
        assert isinstance(config.dataset, NumpyFSLDatasetConfig)
        assert not isinstance(config.dataset, NumpyPackedFSLDatasetConfig)

    def test_use_packing_enables_packed_dataset(self, tmp_path):
        """use_packing: true must switch the pretrain path to NumpyPackedFSLDatasetConfig."""
        yaml_path = _write_minimal_config(tmp_path)
        content = yaml_path.read_text() + "\nuse_packing: true\n"
        yaml_path.write_text(content)

        config = build_experiment_config(config_path=str(yaml_path), run_name="test")
        assert isinstance(config.dataset, NumpyPackedFSLDatasetConfig)

    def test_no_rope_scaling_by_default(self, tmp_path):
        """Without a rope_scaling section, no scaling overrides should be applied."""
        yaml_path = _write_minimal_config(tmp_path)
        config = build_experiment_config(config_path=str(yaml_path), run_name="test")
        assert config.model.block_overrides is None

    def test_rope_scaling_yarn_applied(self, tmp_path):
        """rope_scaling: {type: yarn, ...} in YAML must apply YaRN scaling to full-attn layers."""
        yaml_path = _write_minimal_config(tmp_path)
        content = yaml_path.read_text() + (
            "\nrope_scaling:\n"
            "  type: yarn\n"
            "  factor: 4.0\n"
            "  old_context_len: 2048\n"
        )
        yaml_path.write_text(content)

        config = build_experiment_config(config_path=str(yaml_path), run_name="test")
        # olmo3_190M uses a sliding-window pattern, so scaling is applied via
        # block_overrides on the full-attention layers only, not on the shared
        # base block (which is used for the sliding-window layers).
        assert config.model.block_overrides
        scaling = next(iter(config.model.block_overrides.values())).sequence_mixer.rope.scaling
        assert isinstance(scaling, YaRNRoPEScalingConfig)
        assert scaling.factor == 4.0
        assert scaling.old_context_len == 2048

    def test_rope_scaling_unsupported_type_raises(self, tmp_path):
        """An unsupported rope_scaling type must raise a clear error."""
        yaml_path = _write_minimal_config(tmp_path)
        content = yaml_path.read_text() + "\nrope_scaling:\n  type: linear\n"
        yaml_path.write_text(content)

        with pytest.raises(ValueError, match="Unsupported rope_scaling type"):
            build_experiment_config(config_path=str(yaml_path), run_name="test")


def _with_train_module(yaml_path, extra: dict):
    """Merge `extra` into the train_module section of a YAML config written by
    _write_minimal_config."""
    import yaml

    raw = yaml.safe_load(yaml_path.read_text())
    raw["train_module"].update(extra)
    yaml_path.write_text(yaml.safe_dump(raw))
    return yaml_path


def _build(tmp_path, extra=None, overrides=None):
    yaml_path = _write_minimal_config(tmp_path)
    if extra:
        _with_train_module(yaml_path, extra)
    return build_experiment_config(
        config_path=str(yaml_path), run_name="test", overrides=overrides
    ).train_module


class TestParallelismConfig:
    """TP / PP / CP / AC / float8 / HSDP settings must flow from YAML to the train module."""

    def test_everything_off_by_default(self, tmp_path):
        tm = _build(tmp_path)
        assert tm.tp_config is None
        assert tm.pp_config is None
        assert tm.cp_config is None
        assert tm.ac_config is None
        assert tm.float8_config is None

    def test_default_dp_is_fsdp(self, tmp_path):
        from olmo_core.distributed.parallel import DataParallelType

        tm = _build(tmp_path)
        assert tm.dp_config.name == DataParallelType.fsdp
        assert tm.dp_config.num_replicas is None and tm.dp_config.shard_degree is None

    def test_default_dtypes(self, tmp_path):
        from olmo_core.config import DType

        tm = _build(tmp_path)
        assert tm.dp_config.param_dtype == DType.bfloat16
        assert tm.dp_config.reduce_dtype == DType.float32

    @pytest.mark.parametrize("name", ["fsdp", "hsdp", "ddp"])
    def test_dtypes_flow_from_yaml(self, tmp_path, name):
        from olmo_core.config import DType

        tm = _build(
            tmp_path,
            {"dp_config": {"name": name, "param_dtype": "float32", "reduce_dtype": "bfloat16"}},
        )
        assert tm.dp_config.param_dtype == DType.float32
        assert tm.dp_config.reduce_dtype == DType.bfloat16

    def test_dtypes_from_cli_override(self, tmp_path):
        from olmo_core.config import DType

        tm = _build(
            tmp_path,
            {"dp_config": {"name": "fsdp", "param_dtype": "bfloat16"}},
            overrides=["train_module.dp_config.param_dtype=float32"],
        )
        assert tm.dp_config.param_dtype == DType.float32

    def test_invalid_dtype_raises(self, tmp_path):
        with pytest.raises(ValueError, match="dp_config.param_dtype"):
            _build(tmp_path, {"dp_config": {"param_dtype": "bf17"}})

    def test_hsdp(self, tmp_path):
        from olmo_core.distributed.parallel import DataParallelType

        tm = _build(
            tmp_path,
            {"dp_config": {"name": "hsdp", "num_replicas": 2, "shard_degree": 8}},
        )
        assert tm.dp_config.name == DataParallelType.hsdp
        assert (tm.dp_config.num_replicas, tm.dp_config.shard_degree) == (2, 8)

    def test_hsdp_with_only_shard_degree(self, tmp_path):
        tm = _build(tmp_path, {"dp_config": {"name": "hsdp", "shard_degree": 4}})
        assert tm.dp_config.shard_degree == 4 and tm.dp_config.num_replicas is None

    def test_replicas_without_hsdp_raises(self, tmp_path):
        with pytest.raises(ValueError, match="only valid with name: hsdp"):
            _build(tmp_path, {"dp_config": {"name": "fsdp", "num_replicas": 2}})

    def test_tensor_parallel(self, tmp_path):
        tm = _build(tmp_path, {"tp_config": {"degree": 4, "enable_async": True}})
        assert tm.tp_config.degree == 4
        assert tm.tp_config.enable_async is True

    def test_pipeline_parallel(self, tmp_path):
        from olmo_core.distributed.parallel import PipelineScheduleType

        tm = _build(
            tmp_path,
            {
                "pp_config": {
                    "degree": 2,
                    "schedule": "1F1B",
                    "split_points": [8],
                    "use_custom_stage_implementation": True,
                }
            },
        )
        assert tm.pp_config.degree == 2
        assert tm.pp_config.schedule == PipelineScheduleType.single_1F1B
        assert tm.pp_config.split_points == [8]
        assert tm.pp_config.use_custom_stage_implementation is True

    def test_pipeline_schedule_accepts_enum_name(self, tmp_path):
        from olmo_core.distributed.parallel import PipelineScheduleType

        tm = _build(tmp_path, {"pp_config": {"degree": 2, "schedule": "single_1F1B"}})
        assert tm.pp_config.schedule == PipelineScheduleType.single_1F1B

    def test_context_parallel_ring_defaults(self, tmp_path):
        from olmo_core.nn.attention.ring import RingAttentionLoadBalancerType

        tm = _build(tmp_path, {"cp_config": {"degree": 2}})
        assert tm.cp_config.degree == 2
        assert tm.cp_config.uly is None
        assert tm.cp_config.ring.load_balancer == RingAttentionLoadBalancerType.zig_zag

    def test_context_parallel_llama3(self, tmp_path):
        from olmo_core.nn.attention.ring import RingAttentionLoadBalancerType

        tm = _build(
            tmp_path,
            {"cp_config": {"degree": 4, "load_balancer": "llama3", "head_stride": 2}},
        )
        assert tm.cp_config.ring.load_balancer == RingAttentionLoadBalancerType.llama3
        assert tm.cp_config.ring.head_stride == 2

    def test_context_parallel_ulysses(self, tmp_path):
        tm = _build(tmp_path, {"cp_config": {"degree": 2, "style": "ulysses"}})
        assert tm.cp_config.uly is not None and tm.cp_config.ring is None

    def test_ulysses_rejects_ring_options(self, tmp_path):
        with pytest.raises(ValueError, match="only apply to style: ring"):
            _build(
                tmp_path,
                {"cp_config": {"degree": 2, "style": "ulysses", "head_stride": 2}},
            )

    def test_activation_checkpointing_full(self, tmp_path):
        from olmo_core.nn.transformer import TransformerActivationCheckpointingMode

        tm = _build(tmp_path, {"ac_config": {"mode": "full"}})
        assert tm.ac_config.mode == TransformerActivationCheckpointingMode.full

    def test_activation_checkpointing_selected_blocks(self, tmp_path):
        from olmo_core.nn.transformer import TransformerActivationCheckpointingMode

        tm = _build(
            tmp_path, {"ac_config": {"mode": "selected_blocks", "block_interval": 2}}
        )
        assert tm.ac_config.mode == TransformerActivationCheckpointingMode.selected_blocks
        assert tm.ac_config.block_interval == 2

    def test_activation_checkpointing_selected_blocks_needs_interval(self, tmp_path):
        from olmo_core.exceptions import OLMoConfigurationError

        with pytest.raises(OLMoConfigurationError, match="block_interval"):
            _build(tmp_path, {"ac_config": {"mode": "selected_blocks"}})

    def test_float8(self, tmp_path):
        from olmo_core.float8 import AOFloat8LinearRecipe

        tm = _build(
            tmp_path,
            {"float8_config": {"recipe": "rowwise", "modules_to_ignore": ["lm_head"]}},
        )
        assert tm.float8_config.enabled is True
        assert tm.float8_config.ao_recipe == AOFloat8LinearRecipe.rowwise
        assert tm.float8_config.modules_to_ignore == ["lm_head"]

    def test_float8_can_be_disabled(self, tmp_path):
        tm = _build(tmp_path, {"float8_config": {"enabled": False}})
        assert tm.float8_config.enabled is False

    def test_cli_override_enables_parallelism(self, tmp_path):
        """Dotlist overrides must work on the new sections, e.g. for launcher scripts."""
        tm = _build(
            tmp_path,
            overrides=["train_module.tp_config.degree=2", "train_module.ac_config.mode=full"],
        )
        assert tm.tp_config.degree == 2
        assert tm.ac_config is not None

    @pytest.mark.parametrize("section", ["dp_config", "tp_config", "pp_config", "cp_config", "ac_config", "float8_config"])
    def test_unknown_key_raises(self, tmp_path, section):
        """A typo in a parallelism section must fail loudly, not be silently ignored."""
        body = {"degree": 2} if section in ("tp_config", "pp_config", "cp_config") else {}
        body["definitely_not_a_key"] = 1
        with pytest.raises(ValueError, match=f"Unknown key.*{section}"):
            _build(tmp_path, {section: body})

    @pytest.mark.parametrize("section", ["tp_config", "pp_config", "cp_config"])
    @pytest.mark.parametrize("degree", [0, -1, 1.5, "2", True])
    def test_invalid_degree_raises(self, tmp_path, section, degree):
        with pytest.raises(ValueError, match="positive integer"):
            _build(tmp_path, {section: {"degree": degree}})

    @pytest.mark.parametrize("section", ["tp_config", "pp_config", "cp_config"])
    def test_missing_degree_raises(self, tmp_path, section):
        with pytest.raises(ValueError, match="degree is required"):
            _build(tmp_path, {section: {}})

    def test_invalid_enum_value_lists_choices(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid train_module.pp_config.schedule"):
            _build(tmp_path, {"pp_config": {"degree": 2, "schedule": "nope"}})

    def test_non_mapping_section_raises(self, tmp_path):
        with pytest.raises(ValueError, match="must be a mapping"):
            _build(tmp_path, {"tp_config": 4})

    def test_all_options_together_are_serializable(self, tmp_path):
        """ExperimentConfig.as_config_dict() runs on every launch (W&B + checkpoints), so the
        enum-valued parallelism configs must serialize."""
        yaml_path = _write_minimal_config(tmp_path)
        _with_train_module(
            yaml_path,
            {
                "dp_config": {"name": "hsdp", "num_replicas": 2, "shard_degree": 4},
                "tp_config": {"degree": 2},
                "pp_config": {"degree": 2, "schedule": "1F1B"},
                "cp_config": {"degree": 2, "load_balancer": "llama3"},
                "ac_config": {"mode": "full"},
                "float8_config": {"recipe": "rowwise"},
            },
        )
        config = build_experiment_config(config_path=str(yaml_path), run_name="test")
        d = config.as_config_dict()
        assert d["train_module"]["tp_config"]["degree"] == 2
        assert d["train_module"]["dp_config"]["name"] == "hsdp"


_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
# variant file suffix -> the one feature it must switch on
_7B_VARIANTS = {
    "hsdp": "dp_config",
    "tp": "tp_config",
    "pp": "pp_config",
    "cp": "cp_config",
    "ac": "ac_config",
    "float8": "float8_config",
}
_FEATURES = ("tp_config", "pp_config", "cp_config", "ac_config", "float8_config")


def _7b_train_module(name):
    import yaml

    from t0_training.olmo import config as cfg

    tm = yaml.safe_load((_CONFIGS_DIR / name).read_text())["train_module"]
    return {
        "dp_config": cfg._build_dp_config(tm.get("dp_config", {})),
        "tp_config": cfg._build_tp_config(tm.get("tp_config")),
        "pp_config": cfg._build_pp_config(tm.get("pp_config")),
        "cp_config": cfg._build_cp_config(tm.get("cp_config")),
        "ac_config": cfg._build_ac_config(tm.get("ac_config")),
        "float8_config": cfg._build_float8_config(tm.get("float8_config")),
    }


class TestOlmo3SevenBVariants:
    """configs/olmo3-7B-<variant>.yaml: one parallelism option each, nothing else changed."""

    def test_base_7b_has_no_optional_parallelism(self):
        built = _7b_train_module("olmo3-7B.yaml")
        assert all(built[f] is None for f in _FEATURES)

    @pytest.mark.parametrize("variant,feature", sorted(_7B_VARIANTS.items()))
    def test_variant_enables_only_its_feature(self, variant, feature):
        from olmo_core.distributed.parallel import DataParallelType

        built = _7b_train_module(f"olmo3-7B-{variant}.yaml")
        if feature == "dp_config":
            assert built["dp_config"].name == DataParallelType.hsdp
            assert (built["dp_config"].num_replicas, built["dp_config"].shard_degree) == (4, 4)
            assert all(built[f] is None for f in _FEATURES)
        else:
            assert built[feature] is not None
            assert built["dp_config"].name == DataParallelType.fsdp
            assert all(built[f] is None for f in _FEATURES if f != feature)

    @pytest.mark.parametrize("variant", sorted(_7B_VARIANTS))
    def test_variant_otherwise_matches_base(self, variant):
        """Variants must not drift from the base 7B config in anything but their own section."""
        import yaml

        base = yaml.safe_load((_CONFIGS_DIR / "olmo3-7B.yaml").read_text())
        var = yaml.safe_load((_CONFIGS_DIR / f"olmo3-7B-{variant}.yaml").read_text())
        for d in (base, var):
            d["train_module"].pop("dp_config", None)
            for f in _FEATURES:
                d["train_module"].pop(f, None)
        assert var == base
