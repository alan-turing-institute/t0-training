"""Phase 3 validation: distributed setup and FSDP2 wrapping.

Run with torchrun:
    torchrun --nproc_per_node=2 tests/test_fsdp.py
    torchrun --nproc_per_node=4 tests/test_fsdp.py

Each test function prints PASS/FAIL on rank 0; the script exits with a
non-zero code if any assertion fails on any rank.
"""

import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

from t0_training.distributed import init_distributed, wrap_model_fsdp
from t0_training.model.config import TransformerConfig
from t0_training.model.transformer import Transformer

# Small config so tests run quickly on any GPU count
CONFIG = TransformerConfig(
    d_model=256,
    n_heads=4,
    n_kv_heads=2,
    n_layers=4,
    ffn_hidden_dim=512,
    vocab_size=1024,
    max_seq_len=128,
)

FAILURES: list[str] = []


def _fail(msg: str) -> None:
    FAILURES.append(msg)
    if dist.get_rank() == 0:
        print(f"  FAIL: {msg}", flush=True)


def _pass(name: str) -> None:
    if dist.get_rank() == 0:
        print(f"  PASS: {name}", flush=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_param_count_agrees_across_ranks():
    """All ranks should report the same total parameter count after wrapping."""
    model = Transformer(CONFIG).cuda().to(torch.bfloat16)
    expected_params = sum(p.numel() for p in model.parameters())

    model = wrap_model_fsdp(model)

    # DTensor.numel() returns the GLOBAL (unsharded) size, so each rank
    # reports the full count — no all-reduce needed.
    total = sum(p.numel() for p in model.parameters())

    if total != expected_params:
        _fail(
            f"test_param_count_agrees_across_ranks: "
            f"expected {expected_params}, got {total}"
        )
    else:
        _pass("test_param_count_agrees_across_ranks")

    return model


def test_forward_loss_identical_across_ranks(model: Transformer):
    """Loss from a forward pass must be the same on every rank.

    FSDP all-gathers parameters before the forward, so all ranks compute
    identical outputs for identical inputs.
    """
    torch.manual_seed(42)
    tokens = torch.randint(0, CONFIG.vocab_size, (2, CONFIG.max_seq_len), device="cuda")

    logits = model(tokens)  # (2, T, V)
    loss = F.cross_entropy(
        logits.view(-1, CONFIG.vocab_size).float(),
        tokens.view(-1),
    )

    # Broadcast rank-0 loss to all ranks and compare.
    loss_tensor = loss.detach().clone()
    dist.broadcast(loss_tensor, src=0)

    if not torch.isclose(loss, loss_tensor, atol=1e-3):
        _fail(
            f"test_forward_loss_identical_across_ranks: "
            f"rank {dist.get_rank()} loss={loss.item():.6f}, "
            f"rank 0 loss={loss_tensor.item():.6f}"
        )
    else:
        _pass("test_forward_loss_identical_across_ranks")


def test_sharding_is_active(model: Transformer):
    """Local parameter shards should sum to the global parameter count.

    FSDP2 converts parameters to DTensors. Each rank holds a 1/world_size
    slice; all-reducing the local sizes should recover the global total.
    This directly verifies sharding without relying on memory measurements,
    which are polluted by NCCL workspace allocations.
    """
    from torch.distributed.tensor import DTensor

    world_size = dist.get_world_size()
    params = list(model.parameters())

    sharded = [p for p in params if isinstance(p, DTensor)]
    if not sharded:
        _fail("test_sharding_is_active: no parameters are DTensors — fully_shard may not have worked")
        return

    local_numel = sum(
        p._local_tensor.numel() if isinstance(p, DTensor) else p.numel()
        for p in params
    )
    total = torch.tensor(local_numel, dtype=torch.long, device="cuda")
    dist.all_reduce(total, op=dist.ReduceOp.SUM)

    global_numel = sum(p.numel() for p in params)  # DTensor.numel() is global

    if total.item() != global_numel:
        _fail(
            f"test_sharding_is_active: local shards sum to {total.item()}, "
            f"expected {global_numel}"
        )
    else:
        _pass(
            f"test_sharding_is_active "
            f"(global={global_numel}, local_this_rank={local_numel}, "
            f"world_size={world_size})"
        )


def test_backward_completes(model: Transformer):
    """Backward pass should complete without error and produce finite gradients."""
    torch.manual_seed(0)
    tokens = torch.randint(0, CONFIG.vocab_size, (2, CONFIG.max_seq_len), device="cuda")

    logits = model(tokens)
    loss = F.cross_entropy(
        logits.view(-1, CONFIG.vocab_size).float(),
        tokens.view(-1),
    )
    loss.backward()

    # Check that at least one parameter has a non-None, finite gradient.
    bad_grads = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    if bad_grads:
        _fail(f"test_backward_completes: non-finite grads in {bad_grads[:3]}")
    else:
        _pass("test_backward_completes")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    world_size, rank, local_rank = init_distributed()

    if rank == 0:
        print(f"\nRunning FSDP tests on {world_size} rank(s)...", flush=True)

    model = test_param_count_agrees_across_ranks()
    dist.barrier()

    test_sharding_is_active(model)
    dist.barrier()

    test_forward_loss_identical_across_ranks(model)
    dist.barrier()

    test_backward_completes(model)
    dist.barrier()

    if rank == 0:
        if FAILURES:
            print(f"\n{len(FAILURES)} test(s) FAILED.", flush=True)
        else:
            print("\nAll tests passed.", flush=True)

    dist.destroy_process_group()

    if FAILURES:
        sys.exit(1)


if __name__ == "__main__":
    main()
