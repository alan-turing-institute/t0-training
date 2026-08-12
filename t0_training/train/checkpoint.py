import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

# https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/distributed/checkpoint


def capture_rng_state() -> dict[str, Any]:
    """Snapshot of this rank's RNG state (python, numpy, torch, and cuda if available)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


class CheckpointManager:
    """Saves/restores sharded model + optimizer state via torch.distributed.checkpoint,
    plus an arbitrary caller-supplied "trainer state" dict (step, RNG, data loader
    position, etc.).

    Each rank writes/reads only its own shard -- no gather-to-rank-0 anti-pattern.
    `dcp.save`/`dcp.load` fall back to a single-process mode automatically when no
    process group is initialized, so this class works unchanged for both the real
    multi-node FSDP2 job and single-GPU/CPU testing.

    The trainer state is saved per-rank (via `torch.save`, not JSON) because it
    contains rank-local RNG state that can't be shared across ranks and isn't
    JSON-serializable (torch/numpy RNG states are tensors/structured arrays).
    """

    TRAINER_STATE_FNAME = "trainer_state_rank{rank}.pt"

    def __init__(self, save_dir: str | Path, keep_last_n: int = 3):
        self.save_dir = Path(save_dir)
        self.keep_last_n = keep_last_n

    def _rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    def _ckpt_dir(self, step: int) -> Path:
        return self.save_dir / f"step_{step}"

    def _checkpoint_steps(self) -> list[int]:
        if not self.save_dir.exists():
            return []
        steps = []
        for p in self.save_dir.iterdir():
            if p.is_dir() and p.name.startswith("step_"):
                try:
                    steps.append(int(p.name.removeprefix("step_")))
                except ValueError:
                    continue
        return sorted(steps)

    def latest_checkpoint(self) -> Path | None:
        steps = self._checkpoint_steps()
        return self._ckpt_dir(steps[-1]) if steps else None

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        trainer_state: dict[str, Any],
    ) -> Path:
        ckpt_dir = self._ckpt_dir(step)

        model_state, optim_state = get_state_dict(model, optimizer)
        dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=str(ckpt_dir))

        if self._rank() == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        # Per-rank: RNG state (and anything else rank-local) can't be shared.
        rank = self._rank()
        torch.save(trainer_state, ckpt_dir / self.TRAINER_STATE_FNAME.format(rank=rank))

        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            self._prune_old_checkpoints()

        return ckpt_dir

    def _prune_old_checkpoints(self) -> None:
        if self.keep_last_n <= 0:
            return
        steps = self._checkpoint_steps()
        for step in steps[: -self.keep_last_n]:
            shutil.rmtree(self._ckpt_dir(step), ignore_errors=True)

    def load(
        self,
        checkpoint_dir: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        checkpoint_dir = Path(checkpoint_dir)

        model_state, optim_state = get_state_dict(model, optimizer)
        dcp.load(
            {"model": model_state, "optimizer": optim_state},
            checkpoint_id=str(checkpoint_dir),
        )
        set_state_dict(
            model,
            optimizer,
            model_state_dict=model_state,
            optim_state_dict=optim_state,
        )

        # Prefer this rank's own trainer state; fall back to rank 0's if the
        # checkpoint was written with a different world size.
        rank = self._rank()
        rank_file = checkpoint_dir / self.TRAINER_STATE_FNAME.format(rank=rank)
        if not rank_file.exists():
            rank_file = checkpoint_dir / self.TRAINER_STATE_FNAME.format(rank=0)
        return torch.load(rank_file, weights_only=False)
