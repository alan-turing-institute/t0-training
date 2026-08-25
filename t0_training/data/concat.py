import itertools
from bisect import bisect_right

import torch
from torch.utils.data import Dataset

from .dataset import NumpyDataset


class ConcatNumpyDataset(Dataset):
    """Concatenates multiple NumpyDatasets into a single flat index space.

    Mirrors olmo-core's NumpyFSLDataset: instances are addressed by
    cumulative offset, so a sampler that shuffles range(len(self)) without
    replacement (e.g. GlobalShuffleSampler) sees every instance from every
    file exactly once per epoch, naturally proportional to each file's own
    instance count -- no explicit weights needed.
    """

    def __init__(self, datasets: list[NumpyDataset]):
        assert len(datasets) > 0
        self.datasets = datasets
        # store a list of cumulative lengths of each dataset
        self._cumulative = list(itertools.accumulate(len(ds) for ds in datasets))

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        # bisect_right gives us the index of the dataset that contains the global index `idx`
        ds_idx = bisect_right(self._cumulative, idx)
        # find the local index within that dataset by subtracting the cumulative length of all previous datasets
        local_idx = idx - (self._cumulative[ds_idx - 1] if ds_idx > 0 else 0)
        # get the item using the local index
        return self.datasets[ds_idx][local_idx]
