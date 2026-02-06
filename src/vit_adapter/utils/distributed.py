import os
import builtins

import torch
import torch.distributed as dist


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master: bool):
    builtin_print = builtins.print

    def print(*args, **kwargs):
        if is_master:
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def all_reduce_tensor(tensor: torch.Tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor
