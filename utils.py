'''
    Distributed training related functions.

    From DeiT.
'''

import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

import psutil


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    #print('| distributed init (rank {}): {}'.format(
    #    args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

def equivariance_regulariser(model, where='relaxed-linear', p=2):
    """
    Returns: dict with totals and composed loss scalar
    """
    from nets.relaxed_linear import RelaxedLinearRS  # adjust import if path differs

    eq_norm_total = None
    non_norm_total = None

    for m in model.modules():
        if isinstance(m, RelaxedLinearRS) or (where == 'all' and hasattr(m, 'penalty_terms')):
            eq, non = m.penalty_terms()   # both are norms (float tensors)
            eq = eq if p == 1 else eq**2
            non = non if p == 1 else non**2
            eq_norm_total = eq if eq_norm_total is None else (eq_norm_total + eq)
            non_norm_total = non if non_norm_total is None else (non_norm_total + non)

    if eq_norm_total is None:
        # No modules found; return zeros on model device
        device = next(model.parameters()).device
        zero = torch.tensor(0.0, device=device)
        eq_norm_total, non_norm_total = zero, zero

    return eq_norm_total, non_norm_total

def get_max_num_workers(batch_size, dataset_length):
    """Calculate the maximum number of DataLoader workers given batch size and dataset length.
    Args:
        batch_size (int): Batch size for DataLoader.
        dataset_length (int): Total number of samples in the dataset.
    Returns:
        int: Maximum number of workers that can be used without wasting resources.
    """
    if batch_size <= 0 or dataset_length <= 0:
        return 1
    # Each worker should have at least one batch to process
    max_workers = dataset_length // batch_size
    return max(1, max_workers)

def get_optimal_num_workers(batch_size, dataset_length):
    """Determine the optimal number of DataLoader workers based on max workers and CPU count.
    Args:
        batch_size (int): Batch size for DataLoader.
        dataset_length (int): Total number of samples in the dataset.
    Returns:
        int: Optimal number of workers for DataLoader.
    """
    max_workers = get_max_num_workers(batch_size, dataset_length)
    try:
        cpu_count = len(psutil.Process().cpu_affinity())
    except Exception:
        cpu_count = os.cpu_count() or 1
    # Use the minimum of max_workers and available CPUs, but at least 1
    return max(1, min(max_workers, cpu_count))
