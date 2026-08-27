"""Phase 4 validation: scheduler, optimizer, and trainer."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from t0_training.data import DistributedDataLoader, NumpyDataset
from t0_training.model.config import TransformerConfig
from t0_training.model.transformer import Transformer
from t0_training.optim import build_optimizer
from t0_training.train import TrainingConfig, Trainer, get_lr, set_lr

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# ---------------------------------------------------------------------------
# Shared small configs
# ---------------------------------------------------------------------------

# Model small enough to fit comfortably and run fast
_MODEL_CFG = TransformerConfig(
    d_model=256,
    n_heads=4,
    n_kv_heads=2,
    n_layers=2,
    ffn_hidden_dim=512,
    vocab_size=1024,
    max_seq_len=64,
)

# Training config with grad_accum_steps=1 (global_batch == rank_microbatch)
# so no_sync() is never entered — allows testing without FSDP wrapping.
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
    # Raw headerless uint32 array (.tofile, not np.save) -- matches the real
    # corpus format NumpyDataset reads via np.memmap, despite the ".npy" name.
    tokens = np.random.randint(0, _MODEL_CFG.vocab_size, size=(10_000,), dtype=np.uint32)
    tokens.tofile(str(tmp_path / "tokens.npy"))
    ds = NumpyDataset(tmp_path / "tokens.npy", seq_len=_MODEL_CFG.max_seq_len)
    return DistributedDataLoader(ds, batch_size=1, rank=0, world_size=1, num_workers=0)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_scheduler_zero_at_start():
    lr = get_lr(0, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4)
    assert lr == 0.0


def test_scheduler_linear_warmup():
    lr_50 = get_lr(50, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4)
    assert math.isclose(lr_50, 0.5e-3, rel_tol=1e-6)


def test_scheduler_peak_at_warmup_end():
    lr = get_lr(100, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4)
    assert math.isclose(lr, 1e-3, rel_tol=1e-6)


def test_scheduler_cosine_midpoint():
    # Midway through cosine decay: cos(pi/2)=0, so lr = min + 0.5*(max-min)
    lr = get_lr(550, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4)
    expected = 1e-4 + 0.5 * (1e-3 - 1e-4)
    assert math.isclose(lr, expected, rel_tol=1e-6)


def test_scheduler_floor_after_max_steps():
    lr = get_lr(9999, warmup_steps=100, max_steps=1000, max_lr=1e-3, min_lr=1e-4)
    assert lr == 1e-4


def test_set_lr_updates_all_groups():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(
        [{"params": model.weight}, {"params": model.bias}], lr=1e-3
    )
    set_lr(opt, 5e-4)
    assert all(g["lr"] == 5e-4 for g in opt.param_groups)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def test_build_optimizer_two_param_groups():
    model = torch.nn.Linear(4, 4)
    opt = build_optimizer(model, lr=1e-3)
    assert len(opt.param_groups) == 2


def test_build_optimizer_2d_params_get_weight_decay():
    model = torch.nn.Linear(4, 4)  # weight is 2D, bias is 1D
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    decay_group = opt.param_groups[0]
    assert decay_group["weight_decay"] == 0.1
    # weight matrix should be in decay group
    assert any(p.shape == model.weight.shape for p in decay_group["params"])


def test_build_optimizer_1d_params_no_weight_decay():
    model = torch.nn.Linear(4, 4)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    no_decay_group = opt.param_groups[1]
    assert no_decay_group["weight_decay"] == 0.0
    # bias should be in no-decay group
    assert any(p.shape == model.bias.shape for p in no_decay_group["params"])


def test_build_optimizer_hyperparams():
    model = torch.nn.Linear(4, 4)
    opt = build_optimizer(model, lr=2e-4, betas=(0.9, 0.95), eps=1e-8)
    for group in opt.param_groups:
        assert group["betas"] == (0.9, 0.95)
        assert group["eps"] == 1e-8


# ---------------------------------------------------------------------------
# Trainer (requires GPU for FlashAttention)
# ---------------------------------------------------------------------------


@requires_gpu
def test_trainer_derived_quantities():
    """grad_accum_steps and microbatch_size should be computed correctly."""
    model = Transformer(_MODEL_CFG).cuda().to(torch.bfloat16)
    opt = build_optimizer(model, lr=_TRAIN_CFG.max_lr)
    with tempfile.TemporaryDirectory() as tmp:
        loader = _make_loader(Path(tmp))
        trainer = Trainer(model, opt, loader, _TRAIN_CFG, world_size=1, rank=0)
    assert trainer.grad_accum_steps == 1
    assert trainer.microbatch_size == 1


@requires_gpu
def test_trainer_loss_is_finite():
    """A single training step should produce a finite loss."""
    model = Transformer(_MODEL_CFG).cuda().to(torch.bfloat16)
    opt = build_optimizer(model, lr=_TRAIN_CFG.max_lr)
    with tempfile.TemporaryDirectory() as tmp:
        loader = _make_loader(Path(tmp))
        trainer = Trainer(model, opt, loader, _TRAIN_CFG, world_size=1, rank=0)
        data_iter = iter(loader)
        trainer.optimizer.zero_grad()
        input_ids, labels = next(data_iter)
        input_ids = input_ids.to(trainer.device)
        labels = labels.to(trainer.device)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), labels.view(-1)
        )
        assert torch.isfinite(loss)


@requires_gpu
def test_trainer_loss_decreases():
    """Loss should decrease over a short training run on repeated data."""
    torch.manual_seed(42)
    model = Transformer(_MODEL_CFG).cuda().to(torch.bfloat16)
    opt = build_optimizer(model, lr=_TRAIN_CFG.max_lr)
    with tempfile.TemporaryDirectory() as tmp:
        loader = _make_loader(Path(tmp))
        trainer = Trainer(model, opt, loader, _TRAIN_CFG, world_size=1, rank=0)

        losses = []
        data_iter = iter(loader)
        for _ in range(trainer.grad_accum_steps):
            input_ids, labels = next(data_iter)
        # record initial loss before any training
        with torch.no_grad():
            logits = model(input_ids.to(trainer.device))
            initial_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.to(trainer.device).view(-1),
            ).item()

        trainer.train(start_step=0)

        with torch.no_grad():
            logits = model(input_ids.to(trainer.device))
            final_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.to(trainer.device).view(-1),
            ).item()

        assert final_loss < initial_loss, (
            f"loss did not decrease: initial={initial_loss:.4f} final={final_loss:.4f}"
        )


@requires_gpu
def test_trainer_grad_norm_finite():
    """Grad norm after a backward pass should be a finite positive number."""
    model = Transformer(_MODEL_CFG).cuda().to(torch.bfloat16)
    opt = build_optimizer(model, lr=_TRAIN_CFG.max_lr)
    with tempfile.TemporaryDirectory() as tmp:
        loader = _make_loader(Path(tmp))
        trainer = Trainer(model, opt, loader, _TRAIN_CFG, world_size=1, rank=0)
        data_iter = iter(loader)
        trainer.optimizer.zero_grad()
        input_ids, labels = next(data_iter)
        logits = model(input_ids.to(trainer.device))
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            labels.to(trainer.device).view(-1),
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        assert math.isfinite(grad_norm) and grad_norm > 0


@requires_gpu
def test_trainer_lr_changes_during_warmup():
    """LR should increase during warmup steps."""
    model = Transformer(_MODEL_CFG).cuda().to(torch.bfloat16)
    opt = build_optimizer(model, lr=_TRAIN_CFG.max_lr)
    with tempfile.TemporaryDirectory() as tmp:
        loader = _make_loader(Path(tmp))
        cfg = TrainingConfig(
            max_steps=20,
            global_batch_size=64,
            rank_microbatch_tokens=64,
            seq_len=64,
            warmup_steps=10,
            max_lr=1e-3,
            min_lr=1e-4,
            log_interval=100,
        )
        trainer = Trainer(model, opt, loader, cfg, world_size=1, rank=0)
        lrs = []
        for step in range(cfg.warmup_steps):
            lrs.append(get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr))
        assert lrs == sorted(lrs), "LR should increase monotonically during warmup"


# ---------------------------------------------------------------------------
# z-loss
# ---------------------------------------------------------------------------


def test_z_loss_gather_identity_matches_logsumexp():
    """logsumexp(x) = x_target - log_softmax(x)_target for any class -- this is
    the identity Trainer.train() relies on to compute z-loss via O(N) gathers
    instead of a second O(N, vocab_size) .logsumexp(-1) call."""
    torch.manual_seed(0)
    logits = torch.randn(17, 37)  # arbitrary (N, vocab_size)
    labels = torch.randint(0, 37, (17,))

    direct = logits.logsumexp(-1)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    target_log_prob = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    via_gather = target_logit - target_log_prob

    assert torch.allclose(direct, via_gather, atol=1e-5)


@requires_gpu
def test_trainer_z_loss_runs_and_produces_finite_params():
    """A few training steps with z_loss_multiplier set should run without error
    (exercises the gather-based z-loss path in Trainer.train(), including
    SkipStepAdamW's latest_loss/latest_grad_norm wiring) and leave model
    parameters finite."""
    model = Transformer(_MODEL_CFG).cuda().to(torch.bfloat16)
    opt = build_optimizer(model, lr=_TRAIN_CFG.max_lr)
    with tempfile.TemporaryDirectory() as tmp:
        loader = _make_loader(Path(tmp))
        cfg = TrainingConfig(
            max_steps=5,
            global_batch_size=64,
            rank_microbatch_tokens=64,
            seq_len=64,
            warmup_steps=2,
            max_lr=1e-3,
            min_lr=1e-4,
            log_interval=1,
            z_loss_multiplier=1e-5,
        )
        trainer = Trainer(model, opt, loader, cfg, world_size=1, rank=0)
        trainer.train(start_step=0)

    assert all(torch.isfinite(p).all() for p in model.parameters())
