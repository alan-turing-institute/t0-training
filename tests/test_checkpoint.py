"""Phase 5 validation: checkpoint save/restore of model, optimizer, and data loader."""

from pathlib import Path

import numpy as np
import pytest
import torch

from t0_training.data import DistributedDataLoader, NumpyDataset
from t0_training.model.config import TransformerConfig
from t0_training.model.transformer import Transformer
from t0_training.optim import build_optimizer
from t0_training.train import CheckpointManager, Trainer, TrainingConfig, capture_rng_state

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_MODEL_CFG = TransformerConfig(
    d_model=256,
    n_heads=4,
    n_kv_heads=2,
    n_layers=2,
    ffn_hidden_dim=512,
    vocab_size=1024,
    max_seq_len=64,
)

_TRAIN_CFG = TrainingConfig(
    max_steps=10,
    global_batch_size=64,
    rank_microbatch_tokens=64,
    seq_len=64,
    warmup_steps=5,
    max_lr=1e-3,
    min_lr=1e-4,
    log_interval=5,
)


def _make_loader(tmp_path: Path) -> DistributedDataLoader:
    tmp_path.mkdir(parents=True, exist_ok=True)
    # Seeded locally (not np.random.seed) so callers comparing two loaders built
    # from different tmp_path directories (e.g. an uninterrupted vs. resumed run)
    # get identical underlying token content, regardless of call order or of
    # unrelated tests' global RNG use.
    tokens = np.random.default_rng(0).integers(0, _MODEL_CFG.vocab_size, size=(10_000,), dtype=np.int32)
    np.save(str(tmp_path / "tokens.npy"), tokens)
    ds = NumpyDataset(tmp_path / "tokens.npy", seq_len=_MODEL_CFG.max_seq_len)
    return DistributedDataLoader(ds, batch_size=1, rank=0, world_size=1, num_workers=0)


def _make_trainer_state(loader: DistributedDataLoader, step: int) -> dict:
    """Builds a TrainerStateDict-shaped dict without needing a full Trainer."""
    loader_state = loader.state_dict()
    return {
        "global_step": step,
        "global_train_tokens_seen": step * 64,
        "global_train_petaflops": 0.0,
        "max_steps": 100,
        "data_loader": loader_state,
        "epoch": loader_state["epoch"],
        "world_size": 1,
        "rng": capture_rng_state(),
        "callbacks": {},
    }


# ---------------------------------------------------------------------------
# CPU-only: manager mechanics (dcp falls back to a single-process mode when
# no process group is initialized, so these need no GPU/distributed setup)
# ---------------------------------------------------------------------------


def test_latest_checkpoint_none_when_missing(tmp_path):
    manager = CheckpointManager(tmp_path / "does_not_exist")
    assert manager.latest_checkpoint() is None


def test_latest_checkpoint_none_when_empty(tmp_path):
    manager = CheckpointManager(tmp_path)
    assert manager.latest_checkpoint() is None


def test_save_load_roundtrip_plain_model(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Take one optimizer step so state (exp_avg, exp_avg_sq) is non-empty.
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()

    loader = _make_loader(tmp_path / "data")
    manager = CheckpointManager(tmp_path / "ckpts")
    manager.save(step=10, model=model, optimizer=optimizer, trainer_state=_make_trainer_state(loader, 10))

    saved_weight = model.weight.detach().clone()
    saved_exp_avg = optimizer.state[model.weight]["exp_avg"].detach().clone()

    new_model = torch.nn.Linear(4, 4)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    new_loss = new_model(torch.randn(2, 4)).sum()
    new_loss.backward()
    new_optimizer.step()

    ckpt_dir = manager.latest_checkpoint()
    assert ckpt_dir == tmp_path / "ckpts" / "step_10"

    state = manager.load(ckpt_dir, new_model, new_optimizer)

    assert state["global_step"] == 10
    assert state["max_steps"] == 100
    assert "rng" in state and "callbacks" in state
    assert torch.allclose(new_model.weight, saved_weight)
    assert torch.allclose(new_optimizer.state[new_model.weight]["exp_avg"], saved_exp_avg)


def test_loader_state_restored_through_manager(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = _make_loader(tmp_path / "data")

    data_iter = iter(loader)
    for _ in range(3):
        next(data_iter)
    assert loader.state_dict()["samples_consumed"] == 3

    manager = CheckpointManager(tmp_path / "ckpts")
    manager.save(step=3, model=model, optimizer=optimizer, trainer_state=_make_trainer_state(loader, 3))

    fresh_loader = _make_loader(tmp_path / "data")
    assert fresh_loader.state_dict()["samples_consumed"] == 0

    state = manager.load(manager.latest_checkpoint(), model, optimizer)
    fresh_loader.load_state_dict(state["data_loader"])
    assert fresh_loader.state_dict()["samples_consumed"] == 3
    assert fresh_loader.state_dict()["batches_this_epoch"] == 3


def test_keep_last_n_prunes_old_checkpoints(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = _make_loader(tmp_path / "data")

    manager = CheckpointManager(tmp_path / "ckpts", keep_last_n=2)
    for step in (10, 20, 30):
        manager.save(step=step, model=model, optimizer=optimizer, trainer_state=_make_trainer_state(loader, step))

    remaining = sorted(p.name for p in (tmp_path / "ckpts").iterdir())
    assert remaining == ["step_20", "step_30"]
    assert manager.latest_checkpoint() == tmp_path / "ckpts" / "step_30"


# ---------------------------------------------------------------------------
# GPU: full round trip through the real Transformer + Trainer
# ---------------------------------------------------------------------------


@requires_gpu
def test_trainer_checkpoint_resume_matches_uninterrupted_run(tmp_path):
    """Training 10 steps then resuming from step 5 should match an uninterrupted run.

    Mirrors the Phase 5 validation in the plan: the loss trajectory after resume
    should be identical to what an uninterrupted run would have produced, and the
    data loader should not re-see already-consumed batches.
    """
    torch.manual_seed(0)
    save_dir = tmp_path / "ckpts"

    # Uninterrupted reference run.
    torch.manual_seed(42)
    ref_model = Transformer(_MODEL_CFG).cuda().to(torch.bfloat16)
    ref_opt = build_optimizer(ref_model, lr=_TRAIN_CFG.max_lr)
    ref_loader = _make_loader(tmp_path / "data_ref")
    ref_cfg = TrainingConfig(**{**_TRAIN_CFG.__dict__})
    ref_trainer = Trainer(ref_model, ref_opt, ref_loader, ref_cfg, world_size=1, rank=0)
    ref_trainer.train(start_step=0)
    ref_final_state = {k: v.clone() for k, v in ref_model.state_dict().items()}

    # Interrupted run: train to step 5, checkpoint, "restart", resume to step 10.
    torch.manual_seed(42)
    model = Transformer(_MODEL_CFG).cuda().to(torch.bfloat16)
    opt = build_optimizer(model, lr=_TRAIN_CFG.max_lr)
    loader = _make_loader(tmp_path / "data_resumed")
    cfg = TrainingConfig(
        **{**_TRAIN_CFG.__dict__, "max_steps": 5, "save_dir": str(save_dir), "save_interval": 5}
    )
    trainer = Trainer(model, opt, loader, cfg, world_size=1, rank=0)
    trainer.train(start_step=0)

    resumed_model = Transformer(_MODEL_CFG).cuda().to(torch.bfloat16)
    resumed_opt = build_optimizer(resumed_model, lr=_TRAIN_CFG.max_lr)
    resumed_loader = _make_loader(tmp_path / "data_resumed")
    resumed_cfg = TrainingConfig(
        **{**_TRAIN_CFG.__dict__, "save_dir": str(save_dir), "save_interval": 5}
    )
    resumed_trainer = Trainer(
        resumed_model, resumed_opt, resumed_loader, resumed_cfg, world_size=1, rank=0
    )
    start_step = resumed_trainer.resume_from_latest()
    assert start_step == 5
    assert resumed_loader.state_dict()["samples_consumed"] == loader.state_dict()["samples_consumed"]

    resumed_trainer.train(start_step=start_step)
    resumed_final_state = resumed_model.state_dict()

    for name, ref_tensor in ref_final_state.items():
        assert torch.allclose(ref_tensor, resumed_final_state[name], atol=1e-3), (
            f"mismatch in {name} after resume"
        )
