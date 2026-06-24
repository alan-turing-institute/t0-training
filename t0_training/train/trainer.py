import contextlib
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from t0_training.data.loader import DistributedDataLoader
from t0_training.model.transformer import Transformer
from t0_training.train.scheduler import get_lr, set_lr

# https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/train/trainer.py


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

        # Total global tokens per step (for logging)
        self.tokens_per_step = config.global_batch_size

        # Parameter count for MFU: DTensor.numel() returns the global size
        self.n_params = sum(p.numel() for p in model.parameters())

        self._wandb_init()

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
        # ~6 * N_params FLOPs per token (forward + backward)
        flops_per_token = 6 * self.n_params
        actual_flops_per_gpu = (tokens_per_sec * flops_per_token) / self.world_size
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
                # Suppress gradient all-reduce on all but the final microbatch.
                # FSDP2 exposes no_sync() just like DDP.
                ctx = contextlib.nullcontext() if is_last else self.model.no_sync()
                with ctx:
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
            self.optimizer.step()

            lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr)
            set_lr(self.optimizer, lr)

            if (step + 1) % cfg.log_interval == 0:
                t_now = time.perf_counter()
                elapsed = t_now - t_last
                tokens_per_sec = (self.tokens_per_step * cfg.log_interval) / elapsed
                self._log(step + 1, step_loss, grad_norm, lr, tokens_per_sec)
                t_last = t_now
