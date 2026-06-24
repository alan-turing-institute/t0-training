"""Phase 2 validation: NumpyDataset, DataMixture, DistributedDataLoader."""

import itertools
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from t0_training.data import DataMixture, DistributedDataLoader, NumpyDataset


def _make_npy(path: Path, n_tokens: int = 1_000_000) -> Path:
    arr = np.random.randint(0, 1000, size=(n_tokens,), dtype=np.int32)
    np.save(str(path), arr)
    return path


# ---------------------------------------------------------------------------
# NumpyDataset
# ---------------------------------------------------------------------------


def test_numpy_dataset_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        p = _make_npy(Path(tmp) / "tokens.npy")
        ds = NumpyDataset(p, seq_len=1024)
        assert len(ds) == (1_000_000 - 1) // 1024
        item = ds[0]
        assert item.shape == (1025,)
        assert item.dtype == torch.int64


def test_numpy_dataset_no_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        p = _make_npy(Path(tmp) / "tokens.npy")
        ds = NumpyDataset(p, seq_len=1024)
        a = ds[0]
        b = ds[1]
        # last token of window 0 should equal first token of window 1
        assert a[-1] == b[0]


# ---------------------------------------------------------------------------
# DataMixture
# ---------------------------------------------------------------------------


def test_mixture_returns_correct_shape():
    with tempfile.TemporaryDirectory() as tmp:
        ds1 = NumpyDataset(_make_npy(Path(tmp) / "a.npy"), seq_len=512)
        ds2 = NumpyDataset(_make_npy(Path(tmp) / "b.npy"), seq_len=512)
        mix = DataMixture([ds1, ds2], weights=[0.7, 0.3])
        item = mix[0]
        assert item.shape == (513,)


def test_mixture_samples_proportionally():
    """Over many draws the dataset selection should reflect the weights."""
    with tempfile.TemporaryDirectory() as tmp:
        ds1 = NumpyDataset(_make_npy(Path(tmp) / "a.npy"), seq_len=512)
        ds2 = NumpyDataset(_make_npy(Path(tmp) / "b.npy"), seq_len=512)
        # Patch datasets so we can tell them apart by value range
        ds1._tokens = np.zeros(1_000_001, dtype=np.uint16)
        ds2._tokens = np.ones(1_000_001, dtype=np.uint16)
        mix = DataMixture([ds1, ds2], weights=[9.0, 1.0], seed=0)
        draws = [mix[i][0].item() for i in range(1000)]
        frac_from_ds2 = sum(1 for v in draws if v == 1) / 1000
        assert 0.05 < frac_from_ds2 < 0.20  # ~10% ± noise


# ---------------------------------------------------------------------------
# DistributedDataLoader
# ---------------------------------------------------------------------------


def test_loader_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        ds = NumpyDataset(_make_npy(Path(tmp) / "tokens.npy"), seq_len=1024)
        loader = DistributedDataLoader(ds, batch_size=4, rank=0, world_size=1, num_workers=0)
        input_ids, labels = next(iter(loader))
        assert input_ids.shape == (4, 1024)
        assert labels.shape == (4, 1024)


def test_loader_labels_are_shifted():
    with tempfile.TemporaryDirectory() as tmp:
        ds = NumpyDataset(_make_npy(Path(tmp) / "tokens.npy"), seq_len=1024)
        loader = DistributedDataLoader(ds, batch_size=4, rank=0, world_size=1, num_workers=0)
        input_ids, labels = next(iter(loader))
        # For each sequence, labels[t] == input_ids[t+1] — checked via slicing
        assert (labels[:, :-1] == input_ids[:, 1:]).all()


def test_loader_rank_disjoint():
    """Rank 0 and rank 1 should see different batches."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = NumpyDataset(_make_npy(Path(tmp) / "tokens.npy"), seq_len=1024)
        loader0 = DistributedDataLoader(ds, batch_size=4, rank=0, world_size=2, num_workers=0)
        loader1 = DistributedDataLoader(ds, batch_size=4, rank=1, world_size=2, num_workers=0)
        ids0, _ = next(iter(loader0))
        ids1, _ = next(iter(loader1))
        assert not (ids0 == ids1).all(), "rank 0 and rank 1 should see different batches"


def test_loader_samples_consumed_tracked():
    with tempfile.TemporaryDirectory() as tmp:
        ds = NumpyDataset(_make_npy(Path(tmp) / "tokens.npy"), seq_len=1024)
        loader = DistributedDataLoader(ds, batch_size=4, rank=0, world_size=1, num_workers=0)
        for _ in itertools.islice(loader, 3):
            pass
        assert loader._samples_consumed == 12


def test_loader_state_dict_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        ds = NumpyDataset(_make_npy(Path(tmp) / "tokens.npy"), seq_len=1024)
        loader = DistributedDataLoader(ds, batch_size=4, rank=0, world_size=1, num_workers=0)
        for _ in itertools.islice(loader, 5):
            pass
        state = loader.state_dict()
        assert state["samples_consumed"] == 20
        assert state["epoch"] == 0
        assert state["batches_this_epoch"] == 5
        loader2 = DistributedDataLoader(ds, batch_size=4, rank=0, world_size=1, num_workers=0)
        loader2.load_state_dict(state)
        assert loader2._samples_consumed == 20
        assert loader2._pending_start_idx == 20  # 5 batches × 4


def test_loader_deterministic_shuffle():
    """Same seed + epoch must produce the same index order on both ranks."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = NumpyDataset(_make_npy(Path(tmp) / "tokens.npy"), seq_len=1024)
        # Two loaders with same seed but different ranks — same epoch
        loader_a = DistributedDataLoader(ds, batch_size=4, rank=0, world_size=2, num_workers=0, seed=7)
        loader_b = DistributedDataLoader(ds, batch_size=4, rank=0, world_size=2, num_workers=0, seed=7)
        ids_a, _ = next(iter(loader_a))
        ids_b, _ = next(iter(loader_b))
        assert (ids_a == ids_b).all(), "same seed + epoch should yield identical batches"


def test_loader_different_epoch_different_order():
    """Advancing the epoch must change the shuffle."""
    with tempfile.TemporaryDirectory() as tmp:
        ds = NumpyDataset(_make_npy(Path(tmp) / "tokens.npy"), seq_len=1024)
        loader = DistributedDataLoader(ds, batch_size=4, rank=0, world_size=1, num_workers=0)
        batches_per_epoch = len(ds) // 4  # batches DataLoader produces per epoch
        # Pull first batch of epoch 0, then exhaust the rest via next() — not islice(),
        # which would call __iter__ again and reset the epoch counter.
        epoch0_first, _ = next(iter(loader))
        for _ in range(batches_per_epoch):
            next(loader)
        # The (batches_per_epoch)th next() hits StopIteration, wraps to epoch 1
        assert loader._epoch == 1
        epoch1_first, _ = next(loader)
        assert not (epoch0_first == epoch1_first).all(), "epoch 1 should have a different shuffle"
