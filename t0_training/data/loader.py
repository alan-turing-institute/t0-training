import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class GlobalShuffleSampler(Sampler):
    """Computes a global shuffled permutation seeded by (seed + epoch).

    All ranks use the same RNG and permutation; each rank then strides into it
    to get a non-overlapping subset:

        rank 0 → indices[0 :: world_size]
        rank 1 → indices[1 :: world_size]
        ...

    This ensures every sample appears exactly once per epoch across all ranks,
    the shuffle is deterministic and shared, and mid-epoch resume is possible
    via a start offset.
    """

    def __init__(self, n: int, rank: int, world_size: int, seed: int = 0):
        self._n = n
        self.rank = rank
        self.world_size = world_size
        self._seed = seed
        self._epoch = 0
        self._start_idx = 0  # number of rank-local samples to skip (for resume)

    def set_epoch(self, epoch: int, start_idx: int = 0) -> None:
        self._epoch = epoch
        self._start_idx = start_idx

    def __len__(self) -> int:
        remaining = self._n // self.world_size - self._start_idx
        assert remaining >= 0, (
            f"start_idx ({self._start_idx}) exceeds this rank's samples per epoch "
            f"({self._n // self.world_size}); resume offset is out of range"
        )
        return remaining

    def __iter__(self):
        rng = np.random.default_rng(self._seed + self._epoch)
        # Drop tail so all ranks get equal counts
        n_usable = (self._n // self.world_size) * self.world_size
        indices = rng.permutation(n_usable)
        rank_indices = indices[self.rank :: self.world_size]
        yield from rank_indices[self._start_idx:].tolist()


class DistributedDataLoader:
    """Yields (input_ids, labels) pairs using a pre-computed global shuffle.

    All ranks share the same index permutation (seeded by seed + epoch); each
    rank strides into it for a non-overlapping shard.  Supports exact mid-epoch
    resume via state_dict / load_state_dict.

    Args:
        dataset: Dataset whose __getitem__ returns a (seq_len + 1) token tensor.
        batch_size: per-rank batch size.
        rank: this process's global rank.
        world_size: total number of processes.
        num_workers: DataLoader background worker processes.
        seed: base RNG seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        rank: int,
        world_size: int,
        num_workers: int = 2,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self._seed = seed

        self._sampler = GlobalShuffleSampler(len(dataset), rank, world_size, seed)
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self._sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self._epoch: int = 0
        self._batches_this_epoch: int = 0
        self._samples_consumed: int = 0
        # Pending start offset consumed on next iterator creation (used after resume)
        self._pending_start_idx: int = 0
        self._iter = None

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def _reset_iter(self) -> None:
        """Create a fresh DataLoader iterator for the current epoch."""
        self._sampler.set_epoch(self._epoch, start_idx=self._pending_start_idx)
        self._pending_start_idx = 0
        self._iter = iter(self._loader)

    def __iter__(self) -> "DistributedDataLoader":
        self._reset_iter()
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._iter is None:
            self._reset_iter()
        try:
            batch = next(self._iter)
        except StopIteration:
            # Epoch exhausted — advance and reshuffle
            self._epoch += 1
            self._batches_this_epoch = 0
            self._pending_start_idx = 0
            self._reset_iter()
            batch = next(self._iter)

        input_ids = batch[:, :-1].contiguous()  # (B, seq_len)
        labels = batch[:, 1:].contiguous()      # (B, seq_len)
        self._batches_this_epoch += 1
        self._samples_consumed += input_ids.shape[0]
        return input_ids, labels

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "epoch": self._epoch,
            "batches_this_epoch": self._batches_this_epoch,
            "samples_consumed": self._samples_consumed,
        }

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._batches_this_epoch = state["batches_this_epoch"]
        self._samples_consumed = state["samples_consumed"]
        # Skip already-consumed indices within the current epoch
        self._pending_start_idx = self._batches_this_epoch * self.batch_size
        self._iter = None  # will be created with the right start_idx on next call
