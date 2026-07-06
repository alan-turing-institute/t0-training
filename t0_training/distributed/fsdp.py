import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp import fully_shard

from t0_training.model.block import TransformerBlock
from t0_training.model.transformer import Transformer

# https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/distributed/fsdp.py


def wrap_model_fsdp(model: Transformer) -> Transformer:
    """Wrap a Transformer with FSDP2 (fully_shard).

    Shards parameters, gradients, and optimizer state uniformly across all ranks.
    Each TransformerBlock is sharded individually first to keep per-block peak
    memory low (params are gathered/scattered one block at a time).

    The model must be in fp32 on entry. MixedPrecisionPolicy keeps the sharded
    params in fp32 (the optimizer "master" copy) and casts to bf16 only for the
    forward/backward compute. Pre-casting the model to bf16 before wrapping would
    leave no fp32 master, so the optimizer would update bf16 weights directly —
    defeating mixed precision. Do NOT call .to(torch.bfloat16) before this.

    Args:
        model: Unwrapped fp32 Transformer, already on the correct CUDA device.

    Returns:
        The same model instance, now wrapped with FSDP2.
    """
    param_dtypes = {p.dtype for p in model.parameters()}
    if param_dtypes != {torch.float32}:
        raise ValueError(
            f"wrap_model_fsdp expects an fp32 model so MixedPrecisionPolicy can keep "
            f"an fp32 optimizer master; got param dtypes {param_dtypes}. Remove any "
            f".to(torch.bfloat16) before wrapping — the policy handles the bf16 cast."
        )

    world_size = dist.get_world_size()
    mesh = init_device_mesh("cuda", (world_size,))

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # Shard each block individually — prevents FSDP from concatenating all
    # block params into one giant flat tensor.
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=mp_policy)

    # Shard the full model (embedding, norm, lm_head + already-sharded blocks).
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    return model
