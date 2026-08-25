import os

import torch
import torch.distributed as dist

# https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/distributed/utils.py


def init_distributed() -> tuple[int, int, int]:
    """Initialise NCCL process group and set the CUDA device for this rank.

    Must be called before any model or tensor is created.

    Returns:
        (world_size, rank, local_rank)
    """
    local_rank = int(os.environ["LOCAL_RANK"]) # identify local rank
    torch.cuda.set_device(local_rank) # set which GPU this process will use
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}")) # initialize NCCL process group
    return dist.get_world_size(), dist.get_rank(), local_rank 
