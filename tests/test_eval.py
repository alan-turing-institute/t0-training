"""Unit tests for t0_training/eval/ -- the in-loop perplexity + downstream eval glue.

These test run_lm_eval's/run_downstream_eval's own logic (label shifting/masking,
CE-loss computation, max_steps cap) against fake/synthetic evaluators, not the real
olmo-core perplexity mix or hellaswag task (those need network access and are
exercised by the single-node smoke test instead -- see the eval plan). This keeps the
suite CPU-only and hermetic, matching test_training.py's split between free-standing
math tests and @requires_gpu Trainer tests.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from olmo_core.data.utils import get_labels
from olmo_core.eval import LMEvaluator

from t0_training.data import DistributedDataLoader, NumpyDataset
from t0_training.eval.lm_eval import run_lm_eval
from t0_training.model.config import TransformerConfig
from t0_training.model.transformer import Transformer
from t0_training.optim import build_optimizer
from t0_training.train import Trainer, TrainingConfig

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _fake_model(vocab_size: int):
    """A tiny CPU-only stand-in for Transformer -- avoids the flash-attn/CUDA
    requirement of the real model so run_lm_eval's own logic can be unit tested."""
    torch.manual_seed(0)
    embed = torch.nn.Embedding(vocab_size, 8)
    proj = torch.nn.Linear(8, vocab_size)

    def forward(input_ids: torch.Tensor) -> torch.Tensor:
        return proj(embed(input_ids))

    return forward


class _FakeMetadataBatches(list):
    """Marks these dict batches as plain, non-DataLoaderBase batches so
    Evaluator.__iter__ just iterates them directly (no reset/reshuffle)."""


def _make_batch(vocab_size: int, batch_size: int, seq_len: int, label: str = "test") -> dict:
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    label_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    label_mask[0, -1] = False  # simulate one padded/ignored position
    return {
        "input_ids": input_ids,
        "label_mask": label_mask,
        "metadata": [{"label": label}] * batch_size,
    }


def test_run_lm_eval_matches_manual_ce_loss():
    vocab_size = 16
    model = _fake_model(vocab_size)
    batch = _make_batch(vocab_size, batch_size=2, seq_len=6)
    evaluator = LMEvaluator(
        name="lm", batches=_FakeMetadataBatches([batch]), labels=["test"], device=torch.device("cpu")
    )

    metrics = run_lm_eval(model, evaluator, max_steps=10, device=torch.device("cpu"))

    labels = get_labels(batch)
    logits = model(batch["input_ids"])
    per_token = F.cross_entropy(
        logits.view(-1, vocab_size).float(), labels.view(-1), ignore_index=-100, reduction="none"
    ).view(labels.shape)
    expected_ce = per_token[batch["label_mask"]].mean().item()

    assert math.isclose(metrics["eval/lm/test/CE loss"], expected_ce, rel_tol=1e-5)
    assert math.isclose(metrics["eval/lm/test/PPL"], math.exp(expected_ce), rel_tol=1e-4)


def test_run_lm_eval_respects_max_steps():
    vocab_size = 8
    calls = []

    def counting_model(input_ids: torch.Tensor) -> torch.Tensor:
        calls.append(input_ids)
        return torch.zeros(*input_ids.shape, vocab_size)

    batches = _FakeMetadataBatches(
        _make_batch(vocab_size, batch_size=1, seq_len=4) for _ in range(5)
    )
    evaluator = LMEvaluator(
        name="lm", batches=batches, labels=["test"], device=torch.device("cpu")
    )

    run_lm_eval(counting_model, evaluator, max_steps=2, device=torch.device("cpu"))

    assert len(calls) == 2


def test_run_lm_eval_resets_metrics_between_calls():
    """A second evaluate() call shouldn't accumulate on top of the first."""
    vocab_size = 8
    model = _fake_model(vocab_size)
    batch = _make_batch(vocab_size, batch_size=1, seq_len=4)
    evaluator = LMEvaluator(
        name="lm", batches=_FakeMetadataBatches([batch]), labels=["test"], device=torch.device("cpu")
    )

    first = run_lm_eval(model, evaluator, max_steps=10, device=torch.device("cpu"))
    second = run_lm_eval(model, evaluator, max_steps=10, device=torch.device("cpu"))

    assert math.isclose(first["eval/lm/test/CE loss"], second["eval/lm/test/CE loss"], rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Trainer wiring (requires GPU for the real Transformer)
# ---------------------------------------------------------------------------

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
    tokens = np.random.default_rng(0).integers(0, _MODEL_CFG.vocab_size, size=(10_000,), dtype=np.uint32)
    tokens.tofile(str(tmp_path / "tokens.npy"))
    ds = NumpyDataset(tmp_path / "tokens.npy", seq_len=_MODEL_CFG.max_seq_len)
    return DistributedDataLoader(ds, batch_size=1, rank=0, world_size=1, num_workers=0)


@requires_gpu
def test_trainer_eval_disabled_by_default():
    """eval_interval defaults to None: no evaluator should be built, and no
    network access should be attempted, so plain Trainer construction stays as
    fast/hermetic as it was before evals existed."""
    model = Transformer(_MODEL_CFG).cuda().to(torch.bfloat16)
    opt = build_optimizer(model, lr=_TRAIN_CFG.max_lr)
    with tempfile.TemporaryDirectory() as tmp:
        loader = _make_loader(Path(tmp))
        trainer = Trainer(model, opt, loader, _TRAIN_CFG, world_size=1, rank=0)

    assert trainer.lm_evaluator is None
    assert trainer.downstream_evaluators == []
