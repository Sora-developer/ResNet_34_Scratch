import os

import torch
import torch.distributed as dist


def set_seed(seed: int = 42) -> None:
    """Sets the seed for CPU and CUDA RNGs.
    Args:
        seed (int): The seed value to set for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_ddp(rank: int, world_size: int) -> None:
    """Initializes the process group for single-node multi-GPU training.
    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Tears down the process group."""
    dist.destroy_process_group()
