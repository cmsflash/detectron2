# Copyright (c) Facebook, Inc. and its affiliates.
import os

import torch
import torch.distributed as dist

from detectron2.utils import comm

__all__ = ["launch"]


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def main_worker(main_func, args):
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl" if has_gpu else "gloo",
        init_method="env://",
    )

    # Setup the local process group
    num_gpus_per_machine = torch.cuda.device_count()
    comm.create_local_process_group(num_gpus_per_machine)

    # synchronize to prevent possible timeout after calling init_process_group
    comm.synchronize()

    main_func(*args)


def launch(main_func, args=()):
    """
    Launch the main function using torchrun.
    This function should be called in the main script.
    """
    main_worker(main_func, args)


# This function is no longer needed as torchrun handles process creation
