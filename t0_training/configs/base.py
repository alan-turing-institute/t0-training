import os
from dataclasses import dataclass

from t0_training.model.config import TransformerConfig
from t0_training.olmo.data import resolve_data_paths
from t0_training.train import TrainingConfig


def resolve_mix(
    mix_file: str,
    data_dir: str = "data/npy",
    tokenizer_id: str = "allenai/dolma2-tokenizer",
) -> tuple[list[str], list[float]]:
    """Resolve a mix file (see data/mixes/*.txt) to local .npy paths, raising
    if any shard is missing, plus one DataMixture weight per path.

    Shards vary from ~10KB to ~1GB, so weights must be proportional to shard
    size (a good proxy for token count, since the header is negligible) --
    the DataMixture default of uniform-per-file weight would wildly
    over-sample small shards relative to large ones.
    """
    paths = resolve_data_paths(mix_file, data_dir, tokenizer_id)
    weights = [os.path.getsize(p) for p in paths]
    return paths, weights


@dataclass
class RunConfig:
    """Everything scripts/train.py needs for one run: model shape, training
    hyperparameters, and data sources."""

    model: TransformerConfig
    training: TrainingConfig

    # Paths to pre-tokenized .npy files (see data/dataset.py). A single path
    # trains on that file directly; multiple paths are combined via
    # DataMixture using data_weights (defaults to uniform -- use resolve_mix()
    # to get size-proportional weights instead).
    data_paths: list[str]
    data_weights: list[float] | None = None
    data_seed: int = 0
    num_workers: int = 4
