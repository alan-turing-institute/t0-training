"""Allow running the training CLI via: python -m t0_training / torchrun -m t0_training."""

from dotenv import load_dotenv

load_dotenv()

import torch
import olmo_core.train.utils as _otu


def _get_cuda_or_hip_version():
    """olmo-core's _get_cuda_version asserts torch.version.cuda is not None,
    which fails on ROCm builds where the version lives in torch.version.hip.
    The tuple is only used as a save/restore compatibility check for GPU RNG
    state, so reporting HIP's version is correct and self-consistent.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        version_str = torch.version.cuda or torch.version.hip
        if version_str is None:
            return None
        major, minor = version_str.split(".")[:2]
        return (int(major), int(minor))
    return None


_otu._get_cuda_version = _get_cuda_or_hip_version

from t0_training.olmo.cli import train_main

train_main()
