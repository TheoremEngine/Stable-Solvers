import torch
import torch.distributed as dist

__all__ = [
    'broadcast_across_processes_', 'distributed_dataset_wrapper',
    'sum_across_processes_',
]


def broadcast_across_processes_(tensor: torch.Tensor,
                                broadcast_rank: int = 0) -> torch.Tensor:
    '''
    Convenience function for broadcasting tensors across processes. This is an
    in-place operation.
    '''
    # TODO: Improve error checking etc.
    if not dist.is_initialized():
        return tensor
    dist.broadcast(tensor, broadcast_rank)
    return tensor


def distributed_dataset_wrapper(ds: torch.utils.data.Dataset) \
        -> torch.utils.data.Dataset:
    '''
    Wraps a :class:`torch.utils.data.Dataset` to split it across processes, as
    well as doing basic checks.
    '''
    if not len(ds):
        raise ValueError('Datasets must have length')
    if not dist.is_initialized():
        return ds

    rank, num_procs = dist.get_rank(), dist.get_world_size()
    idxs = [i for i in range(len(ds)) if i % num_procs == rank]
    return torch.utils.data.Subset(ds, idxs)


def sum_across_processes_(tensor: torch.Tensor) -> torch.Tensor:
    '''
    Convenience function for summing tensors across processes. This is an
    in-place operation.
    '''
    # TODO: Improve error checking etc.
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor)
    return tensor
