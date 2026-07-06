from dataclasses import dataclass

from t0_training.model.config import TransformerConfig
from t0_training.train import TrainingConfig


@dataclass
class RunConfig:
    """Everything scripts/train.py needs for one run: model shape, training
    hyperparameters, and data sources."""

    model: TransformerConfig
    training: TrainingConfig

    # Paths to pre-tokenized .npy files (see data/dataset.py). A single path
    # trains on that file directly; multiple paths are combined via
    # DataMixture using data_weights (defaults to uniform).
    data_paths: list[str]
    data_weights: list[float] | None = None
    data_seed: int = 0
    num_workers: int = 4
