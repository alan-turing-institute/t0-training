"""Training loop for t0-training experiments."""

import logging
from typing import cast

import rich

from olmo_core.distributed.utils import get_rank
from olmo_core.train.callbacks import ConfigSaverCallback
from olmo_core.utils import seed_all

from .config import ExperimentConfig

log = logging.getLogger(__name__)


def train(config: ExperimentConfig):
    if get_rank() == 0:
        rich.print(config)

    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset, dp_process_group=train_module.dp_process_group
    )
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # If we have a load path set and there is no checkpoint in the save folder, load the
    # checkpoint from the load path.
    if (
        not trainer.no_checkpoints
        and not trainer.maybe_load_checkpoint()
        and config.load_path
    ):
        log.info(
            f"Loading checkpoint from {config.load_path} since no checkpoints were found in the save folder..."
        )
        trainer.load_checkpoint(
            config.load_path, load_trainer_state=config.load_trainer_state
        )

    # Train.
    trainer.fit()
