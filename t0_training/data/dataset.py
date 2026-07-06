from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    """Memory-mapped dataset over a flat .npy file of pre-tokenized token IDs.

    The file is never fully loaded into RAM; the OS pages in only the needed
    slices.  Each item is a contiguous run of (seq_len + 1) tokens so the
    caller can form input_ids / labels as tokens[:-1] and tokens[1:].
    """

    def __init__(self, path: str | Path, seq_len: int):
        path = Path(path)
        # Allow_pickle=False is a safety guard; our files are plain integer arrays.
        # Keep the memmap as-is: calling .astype() here would materialise the whole
        # file into RAM and defeat the point of mmap. The stored dtype must already
        # hold the full vocab (e.g. uint32 for a 128k vocab — uint16 would silently
        # wrap token IDs > 65535). We cast each slice to int64 in __getitem__.
        self._tokens = np.load(str(path), mmap_mode="r", allow_pickle=False)
        self.seq_len = seq_len
        # Number of complete (seq_len + 1) windows
        self._n = (len(self._tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        chunk = self._tokens[start : start + self.seq_len + 1]
        # Copy the small slice out of the mmap and widen to int64 for embedding lookup.
        return torch.from_numpy(np.ascontiguousarray(chunk).astype(np.int64))
