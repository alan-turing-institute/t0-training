"""Failing regression tests for SFT readiness issues identified in the audit.

These tests intentionally encode expected SFT behavior that is not implemented yet.
They should fail until the SFT support changes are added.
"""

from pathlib import Path
import tomllib

import numpy as np

from olmo_core.data import NumpyPackedFSLDatasetConfig
from olmo_core.optim import LinearWithWarmup

from t0_training.config import build_experiment_config


def _write_base_config(tmp_path: Path, train_module_overrides: str = "") -> Path:
    """Create a minimal config + fake data files so config construction can run offline."""
    npy_dir = tmp_path / "npy"
    npy_dir.mkdir()
    fake_npy = npy_dir / "fake.npy"
    np.save(str(fake_npy), np.array([1, 2, 3], dtype=np.uint16))

    mix_file = tmp_path / "mix.txt"
    mix_file.write_text("fake,fake.npy\n")

    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    np.save(sft_dir / "token_ids_part_0000.npy", np.array([1, 2, 3, 4], dtype=np.uint16))
    np.save(sft_dir / "labels_mask_part_0000.npy", np.array([1, 0, 1, 0], dtype=np.bool_))

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
  global_batch_size: 32768

train_module:
  compile_model: false
  rank_microbatch_size: 16384
  optim:
    lr: 5.0e-5
  scheduler:
    warmup_steps: 50
{train_module_overrides}

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


def test_sft_data_dir_uses_packed_dataset_and_label_masks(tmp_path: Path):
    """SFT mode should switch dataset config to packed + label masks from sft_data_dir."""
    yaml_path = _write_base_config(tmp_path)
    sft_dir = tmp_path / "sft"

    config = build_experiment_config(
        config_path=str(yaml_path),
        run_name="sft-regression",
        overrides=[f"sft_data_dir={sft_dir}"],
    )

    assert isinstance(config.dataset, NumpyPackedFSLDatasetConfig)
    assert config.dataset.expand_glob
    assert config.dataset.paths == [f"{sft_dir}/token_ids_part_*.npy"]
    assert config.dataset.label_mask_paths == [f"{sft_dir}/labels_mask_part_*.npy"]


def test_sft_optimizer_fields_weight_decay_and_betas_are_respected(tmp_path: Path):
    """SFT optimizer settings from YAML should flow through to AdamWConfig."""
    yaml_path = _write_base_config(
        tmp_path,
        train_module_overrides=(
            "  optim:\n"
            "    lr: 5.0e-5\n"
            "    weight_decay: 0.0\n"
            "    betas: [0.9, 0.95]\n"
        ),
    )

    config = build_experiment_config(config_path=str(yaml_path), run_name="sft-optim")

    assert config.train_module.optim.weight_decay == 0.0
    assert config.train_module.optim.betas == (0.9, 0.95)


def test_sft_linear_scheduler_is_respected(tmp_path: Path):
    """SFT config should allow selecting linear-with-warmup scheduler."""
    yaml_path = _write_base_config(
        tmp_path,
        train_module_overrides=(
            "  scheduler:\n"
            "    name: linear_with_warmup\n"
            "    warmup_steps: 50\n"
            "    alpha_f: 0.0\n"
        ),
    )

    config = build_experiment_config(config_path=str(yaml_path), run_name="sft-scheduler")

    assert isinstance(config.train_module.scheduler, LinearWithWarmup)
    assert config.train_module.scheduler.warmup == 50


def test_pyproject_registers_t0_convert_sft_script():
    """Packaging should expose a converter CLI entrypoint for SFT data conversion."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    scripts = data["project"]["scripts"]
    assert "t0-convert-sft" in scripts
    assert scripts["t0-convert-sft"] == "t0_training.cli:convert_sft_main"


def test_cli_exposes_convert_sft_main():
    """CLI module should define a dedicated entrypoint for SFT conversion."""
    from t0_training import cli

    assert hasattr(cli, "convert_sft_main")
