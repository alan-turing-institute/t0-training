"""Tests for build_experiment_config, focusing on trainer.max_duration support."""

import numpy as np
import pytest

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
