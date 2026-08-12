"""Phase 2 validation: NumpyDataset, ConcatNumpyDataset, DistributedDataLoader."""

import itertools
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from t0_training.data import ConcatNumpyDataset, DistributedDataLoader, NumpyDataset
from t0_training.data.loader import GlobalShuffleSampler


def _make_npy(path: Path, n_tokens: int = 1_000_000) -> Path:
    # Raw headerless uint32 array (.tofile, not np.save) -- matches the real
    # corpus format NumpyDataset reads via np.memmap, despite the ".npy" name.
    arr = np.random.randint(0, 1000, size=(n_tokens,), dtype=np.uint32)
    arr.tofile(str(path))
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
# ConcatNumpyDataset
# ---------------------------------------------------------------------------


def test_concat_dataset_length_is_sum_of_parts():
    with tempfile.TemporaryDirectory() as tmp:
        ds1 = NumpyDataset(_make_npy(Path(tmp) / "a.npy", n_tokens=500_000), seq_len=512)
        ds2 = NumpyDataset(_make_npy(Path(tmp) / "b.npy", n_tokens=1_000_000), seq_len=512)
        concat = ConcatNumpyDataset([ds1, ds2])
        assert len(concat) == len(ds1) + len(ds2)


def test_concat_dataset_indexes_into_correct_file():
    with tempfile.TemporaryDirectory() as tmp:
        ds1 = NumpyDataset(_make_npy(Path(tmp) / "a.npy"), seq_len=512)
        ds2 = NumpyDataset(_make_npy(Path(tmp) / "b.npy"), seq_len=512)
        ds1._tokens = np.zeros(1_000_001, dtype=np.uint16)
        ds2._tokens = np.ones(1_000_001, dtype=np.uint16)
        concat = ConcatNumpyDataset([ds1, ds2])
        # Last instance of ds1, first instance of ds2, straddling the boundary.
        assert concat[len(ds1) - 1][0].item() == 0
        assert concat[len(ds1)][0].item() == 1
        assert concat[len(concat) - 1][0].item() == 1
        with pytest.raises(IndexError):
            concat[len(concat)]


def test_concat_dataset_covers_every_instance_exactly_once_per_epoch():
    """Sampling range(len) without replacement (as GlobalShuffleSampler does)
    must touch every instance from every file exactly once -- no skips, no
    repeats."""
    with tempfile.TemporaryDirectory() as tmp:
        ds1 = NumpyDataset(_make_npy(Path(tmp) / "a.npy", n_tokens=50_000), seq_len=512)
        ds2 = NumpyDataset(_make_npy(Path(tmp) / "b.npy", n_tokens=80_000), seq_len=512)
        concat = ConcatNumpyDataset([ds1, ds2])
        sampler = GlobalShuffleSampler(len(concat), rank=0, world_size=1, seed=0)
        sampler.set_epoch(0)
        visited = sorted(sampler)
        assert visited == list(range(len(concat)))


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
