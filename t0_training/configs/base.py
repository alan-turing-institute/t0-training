from dataclasses import dataclass

from t0_training.model.config import TransformerConfig
from t0_training.olmo.data import resolve_data_paths
from t0_training.train import TrainingConfig


def resolve_mix(
    mix_file: str,
    data_dir: str = "data/npy",
    tokenizer_id: str = "allenai/dolma2-tokenizer",
) -> list[str]:
    """Resolve a mix file (see data/mixes/*.txt) to local .npy shard paths,
    raising if any shard is missing. Pass the result as RunConfig.data_paths:
    scripts/train.py concatenates the shards via ConcatNumpyDataset, so each
    one contributes proportionally to its own instance count -- matching
    olmo-core's NumpyFSLDataset.
    """
    return resolve_data_paths(mix_file, data_dir, tokenizer_id)


@dataclass
class RunConfig:
    """Everything scripts/train.py needs for one run: model shape, training
    hyperparameters, and data sources."""

    model: TransformerConfig
    training: TrainingConfig

    # Paths to pre-tokenized .npy files (see data/dataset.py). A single path
    # trains on that file directly; multiple paths are combined via
    # ConcatNumpyDataset (no-replacement, proportional to each file's own
    # instance count, matching olmo-core).
    data_paths: list[str]
    data_seed: int = 0
    num_workers: int = 4
