import time
from dataclasses import dataclass, field
from typing import Any, TypedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from t0_training.data.loader import DistributedDataLoader
from t0_training.model.transformer import Transformer
from t0_training.train.checkpoint import CheckpointManager, capture_rng_state, restore_rng_state
from t0_training.train.scheduler import get_lr, set_lr

# https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/train/trainer.py


class TrainerStateDict(TypedDict):
    global_step: int
    global_train_tokens_seen: int
    global_train_petaflops: float
    max_steps: int | None
    data_loader: dict[str, Any]
    epoch: int
    world_size: int
    rng: dict[str, Any]
    callbacks: dict[str, dict[str, Any]]


@dataclass
class TrainingConfig:
    # Run length
    max_steps: int

    # Batch / sequence dimensions
    # global_batch_size (tokens) = rank_microbatch_tokens * grad_accum_steps * world_size
    global_batch_size: int = 262_144
    rank_microbatch_tokens: int = 16_384
    seq_len: int = 4_096

    # LR schedule
    warmup_steps: int = 2_000
    max_lr: float = 3e-4
    min_lr: float = 3e-5

    # Regularisation
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Logging
    log_interval: int = 10
    # H100 SXM5 bf16 dense ~989 TFLOP/s; H100 NVL ~835 TFLOP/s
    peak_flops_per_gpu: float = 989e12
    wandb_project: str | None = None

    # Checkpointing
    save_dir: str | None = None
    save_interval: int = 500
    keep_last_n_checkpoints: int = 3


class Trainer:
    def __init__(
        self,
        model: Transformer,
        optimizer: torch.optim.Optimizer,
        loader: DistributedDataLoader,
        config: TrainingConfig,
        world_size: int,
        rank: int,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.config = config
        self.world_size = world_size
        self.rank = rank

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")

        # Microbatch size in sequences; grad_accum_steps derived from config
        tokens_per_rank_per_step = config.global_batch_size // world_size
        self.microbatch_size = config.rank_microbatch_tokens // config.seq_len
        self.grad_accum_steps = tokens_per_rank_per_step // config.rank_microbatch_tokens

        # The loop trusts each next(loader) to yield exactly microbatch_size
        # sequences; if the loader's batch size differs, tokens_per_step, MFU,
        # and the grad-accum math are all silently wrong. Enforce the invariant.
        if self.microbatch_size < 1:
            raise ValueError(
                f"rank_microbatch_tokens ({config.rank_microbatch_tokens}) must be "
                f">= seq_len ({config.seq_len})"
            )
        if loader.batch_size != self.microbatch_size:
            raise ValueError(
                f"loader.batch_size ({loader.batch_size}) must equal microbatch_size "
                f"({self.microbatch_size} = rank_microbatch_tokens // seq_len)"
            )
        if tokens_per_rank_per_step % config.rank_microbatch_tokens != 0:
            raise ValueError(
                f"global_batch_size // world_size ({tokens_per_rank_per_step}) must be "
                f"divisible by rank_microbatch_tokens ({config.rank_microbatch_tokens})"
            )

        # Total global tokens per step (for logging)
        self.tokens_per_step = config.global_batch_size

        # Parameter count for MFU: DTensor.numel() returns the global size
        self.n_params = sum(p.numel() for p in model.parameters())
        self.flops_per_token = 6 * self.n_params  # forward + backward, dense FLOPs

        self.global_step = 0
        self.global_train_tokens_seen = 0
        self.global_train_petaflops = 0.0

        self.checkpoint_manager = (
            CheckpointManager(config.save_dir, keep_last_n=config.keep_last_n_checkpoints)
            if config.save_dir is not None
            else None
        )

        self._wandb_init()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> TrainerStateDict:
        loader_state = self.loader.state_dict()
        return {
            "global_step": self.global_step,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "global_train_petaflops": self.global_train_petaflops,
            "max_steps": self.config.max_steps,
            "data_loader": loader_state,
            "epoch": loader_state["epoch"],
            "world_size": self.world_size,
            "rng": capture_rng_state(),
            # No callback system yet; kept for shape-compatibility with future callbacks.
            "callbacks": {},
        }

    def load_state_dict(self, state: TrainerStateDict) -> None:
        self.loader.load_state_dict(state["data_loader"])
        self.global_step = state["global_step"]
        self.global_train_tokens_seen = state["global_train_tokens_seen"]
        self.global_train_petaflops = state["global_train_petaflops"]
        if state["world_size"] == self.world_size:
            restore_rng_state(state["rng"])

    def resume_from_latest(self) -> int:
        """Load the most recent checkpoint under `save_dir`, if any.

        Returns the step to resume training from (0 if no checkpoint exists).
        """
        if self.checkpoint_manager is None:
            return 0
        ckpt_dir = self.checkpoint_manager.latest_checkpoint()
        if ckpt_dir is None:
            return 0
        state = self.checkpoint_manager.load(ckpt_dir, self.model, self.optimizer)
        self.load_state_dict(state)
        return self.global_step

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _wandb_init(self) -> None:
        self._use_wandb = False
        if self.config.wandb_project and self.rank == 0:
            try:
                import wandb
                wandb.init(project=self.config.wandb_project)
                self._use_wandb = True
            except ImportError:
                pass

    def _log(self, step: int, loss: float, grad_norm: float, lr: float, tokens_per_sec: float) -> None:
        if self.rank != 0:
            return

        # MFU: actual FLOP/s per GPU vs peak FLOP/s
        actual_flops_per_gpu = (tokens_per_sec * self.flops_per_token) / self.world_size
        mfu = actual_flops_per_gpu / self.config.peak_flops_per_gpu

        print(
            f"step={step:6d}  loss={loss:.4f}  grad_norm={grad_norm:.3f}  "
            f"lr={lr:.2e}  tok/s={tokens_per_sec:.0f}  MFU={mfu:.1%}",
            flush=True,
        )

        if self._use_wandb:
            import wandb
            wandb.log({
                "train/loss": loss,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "train/tokens_per_sec": tokens_per_sec,
                "train/mfu": mfu,
            }, step=step)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, start_step: int = 0) -> None:
        cfg = self.config
        self.global_step = start_step
        data_iter = iter(self.loader)
        t_last = time.perf_counter()

        for step in range(start_step, cfg.max_steps):
            self.optimizer.zero_grad()

            step_loss = 0.0
            for micro_step in range(self.grad_accum_steps):
                input_ids, labels = next(data_iter)
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                is_last = (micro_step == self.grad_accum_steps - 1)
                # Suppress the gradient reduce-scatter on all but the final
                # microbatch. FSDP2 (fully_shard) has no no_sync() context manager
                # like DDP/FSDP1; it uses set_requires_gradient_sync() instead.
                # No-op on plain (unsharded) modules used in single-GPU tests.
                if hasattr(self.model, "set_requires_gradient_sync"):
                    self.model.set_requires_gradient_sync(is_last)
                logits = self.model(input_ids)          # (B, T, V)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    labels.view(-1),
                    ignore_index=-1,
                )
                # Divide before accumulating so the sum equals the mean
                (loss / self.grad_accum_steps).backward()

                step_loss += loss.item() / self.grad_accum_steps

            grad_norm = clip_grad_norm_(self.model.parameters(), cfg.grad_clip).item()

            # Set the LR for THIS step before stepping. Doing it after would make
            # optimizer.step() use the previous iteration's LR (and the very first
            # step would use the optimizer's construction LR, skipping warmup).
            lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr)
            set_lr(self.optimizer, lr)
            self.optimizer.step()

            self.global_step = step + 1
            self.global_train_tokens_seen += self.tokens_per_step
            self.global_train_petaflops += (self.flops_per_token * self.tokens_per_step) / 1e15

            if (step + 1) % cfg.log_interval == 0:
                t_now = time.perf_counter()
                elapsed = t_now - t_last
                tokens_per_sec = (self.tokens_per_step * cfg.log_interval) / elapsed
                self._log(step + 1, step_loss, grad_norm, lr, tokens_per_sec)
                t_last = t_now

            if self.checkpoint_manager is not None and (step + 1) % cfg.save_interval == 0:
                self.checkpoint_manager.save(step + 1, self.model, self.optimizer, self.state_dict())
