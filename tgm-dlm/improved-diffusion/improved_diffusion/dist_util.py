"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 1 #8

SETUP_RETRY_COUNT = 3
_DIST_DEVICE = None
_DIST_BACKEND = None


def _cuda_ready(gpu_id):
    try:
        if not th.cuda.is_available():
            return False
        th.cuda.set_device(gpu_id)
        th.empty(1, device=f"cuda:{gpu_id}")
        return True
    except Exception:
        return False


def setup_dist(rank, world_size, port="12145", backend=None, force_cuda=False, gpu_id=0):
    """
    Setup a distributed process group.
    """
    global _DIST_DEVICE, _DIST_BACKEND
    if dist.is_initialized():
        return

    # comm = MPI.COMM_WORLD
    # backend = "gloo" if not th.cuda.is_available() else "nccl"

    # if backend == "gloo":
    #     hostname = "localhost"
    # else:
    #     hostname = socket.gethostbyname(socket.getfqdn())
    # os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    # os.environ["RANK"] = str(comm.rank)
    # os.environ["WORLD_SIZE"] = str(comm.size)

    # port = comm.bcast(_find_free_port(), root=0)
    # os.environ["MASTER_PORT"] = str(port)

    # dist.init_process_group(backend=backend, init_method="env://")
    use_cuda = _cuda_ready(gpu_id)
    if not use_cuda and not force_cuda and backend is None:
        try:
            th.cuda.set_device(gpu_id)
            th.cuda.init()
            th.empty(1, device=f"cuda:{gpu_id}")
            use_cuda = True
        except Exception:
            use_cuda = False
    if force_cuda and not use_cuda:
        try:
            th.cuda.set_device(gpu_id)
            th.cuda.init()
            th.empty(1, device=f"cuda:{gpu_id}")
            use_cuda = True
        except Exception as exc:
            raise RuntimeError(
                f"force_cuda=True but CUDA init failed on gpu_id={gpu_id}: {exc}"
            ) from exc

    if backend is None:
        backend = "nccl" if use_cuda else "gloo"
    if backend == "nccl" and not use_cuda:
        raise RuntimeError(
            f"backend=nccl requested but CUDA is unavailable (gpu_id={gpu_id})"
        )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = str(port)
    if backend == "nccl":
        th.cuda.set_device(gpu_id)
        _DIST_DEVICE = th.device(f"cuda:{gpu_id}")
    else:
        _DIST_DEVICE = th.device("cpu")
    _DIST_BACKEND = backend
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if _DIST_DEVICE is not None:
        return _DIST_DEVICE
    if th.cuda.is_available():
        if MPI is None:
            return th.device("cuda:0")
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    if MPI is not None:
        data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
