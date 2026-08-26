"""Top-level pretraining entry point.

Usage (single node, 4 GPUs):
    torchrun --nproc_per_node=4 scripts/train.py --config t0_training/configs/config_3b.py

Usage (2 nodes x 4 GPUs, via scripts/launch.sh): see that script.
"""

import argparse
import importlib.util
import os
from pathlib import Path

from t0_training.configs.base import RunConfig
from t0_training.data import ConcatNumpyDataset, DistributedDataLoader, NumpyDataset
from t0_training.distributed import compile_model, init_distributed, wrap_model_fsdp
from t0_training.model.transformer import Transformer
from t0_training.optim import build_optimizer
from t0_training.train import Trainer


def load_config(config_path: str) -> RunConfig:
    """Import a config module (e.g. t0_training/configs/config_3b.py) and
    return its RUN_CONFIG. Config files are plain Python so 3B -> 7B is a
    one-line import change, no CLI schema needed."""
    path = Path(config_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RUN_CONFIG


def build_data_loader(config: RunConfig, rank: int, world_size: int) -> DistributedDataLoader:
    # Use the training seq_len (may be shorter than the model's max_seq_len,
    # e.g. 2048 vs 4096) so microbatch token math stays consistent.
    datasets = [NumpyDataset(p, seq_len=config.training.seq_len) for p in config.data_paths]
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        # Concatenate and let the global shuffle sampler draw without replacement,
        # exactly once per epoch per instance, matching olmo-core's NumpyFSLDataset
        # (each file naturally contributes proportionally to its own instance count).
        dataset = ConcatNumpyDataset(datasets)

    microbatch_size = config.training.rank_microbatch_tokens // config.training.seq_len
    return DistributedDataLoader(
        dataset,
        batch_size=microbatch_size,
        rank=rank,
        world_size=world_size,
        num_workers=config.num_workers,
        seed=config.data_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a config module, e.g. t0_training/configs/config_3b.py")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Subdirectory of the config's save_dir to save checkpoints to (e.g. 'run1'), "
        "so multiple runs from the same config don't collide. Also used to name the "
        "wandb run (e.g. 'run1' -> 't0-3b-run1').",
    )
    args = parser.parse_args()

    world_size, rank, local_rank = init_distributed()
    config = load_config(args.config)
    if args.run_name is not None:
        config.training.run_name = args.run_name
        if config.training.save_dir is not None:
            config.training.save_dir = os.path.join(config.training.save_dir, args.run_name)

    # Keep the model in fp32 on entry: wrap_model_fsdp's MixedPrecisionPolicy
    # casts to bf16 for compute and keeps the fp32 sharded params as the
    # optimizer master. Do NOT call .to(torch.bfloat16) before wrapping.
    model = Transformer(config.model).cuda()
    # Compile before FSDP wrapping (matches olmo-core's apply_compile -> apply_fsdp
    # order) -- compiling after would trace through DTensor all-gather/reshard ops
    # instead of just each block's own computation.
    model = compile_model(model)
    model = wrap_model_fsdp(model)

    optimizer = build_optimizer(
        model,
        lr=config.training.max_lr,
        weight_decay=config.training.weight_decay,
    )
    loader = build_data_loader(config, rank, world_size)

    if config.training.max_steps is None:
        # If no max steps in the config, we calculate one epoch of the actual dataset
        # measured from the real (memmap'd) shard sizes at runtime.
        total_tokens = len(loader.dataset) * config.training.seq_len
        config.training.max_steps = total_tokens // config.training.global_batch_size

    trainer = Trainer(model, optimizer, loader, config.training, world_size=world_size, rank=rank)

    start_step = trainer.resume_from_latest()
    trainer.train(start_step=start_step)


if __name__ == "__main__":
    main()
