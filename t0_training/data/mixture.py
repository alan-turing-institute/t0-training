import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import NumpyDataset


class DataMixture(Dataset):
    """Combines multiple NumpyDatasets with sampling weights.

    On each draw a dataset is chosen proportional to its weight, then a random
    position within that dataset is sampled.  This is suitable for mixing
    sources (e.g. 70% web, 20% books, 10% code) without interleaving files.

    Args:
        datasets: list of NumpyDataset instances.
        weights: unnormalised sampling weights, one per dataset.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        datasets: list[NumpyDataset],
        weights: list[float],
        seed: int = 42,
    ):
        assert len(datasets) == len(weights) and len(datasets) > 0
        self.datasets = datasets
        weights_arr = np.array(weights, dtype=np.float64)
        self._probs = weights_arr / weights_arr.sum()
        self._rng = np.random.default_rng(seed)

        # Total length is the weighted harmonic mean of per-dataset sizes,
        # scaled so that every dataset is seen roughly proportionally.
        # In practice, just expose the largest dataset's length — callers
        # should use an IterableDataset wrapper or infinite sampler for real
        # training; this __len__ is used for compatibility with DataLoader.
        self._len = max(len(ds) for ds in datasets)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> torch.Tensor:
        # idx is ignored; we always sample according to the mixture weights.
        ds_idx = int(self._rng.choice(len(self.datasets), p=self._probs))
        ds = self.datasets[ds_idx]
        pos = int(self._rng.integers(0, len(ds)))
        return ds[pos]
