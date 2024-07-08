from math import prod, sqrt
from typing import Callable, Dict
import warnings

import numpy as np
import torch

from .distributed import distributed_dataset_wrapper, sum_across_processes_

__all__ = ['LossFunction']


class LossFunction(torch.nn.Module):
    '''
    This class encapsulates a dataset, criterion, and network architecture, and
    takes as input a set of network parameters. It is intended to allow easy
    encapsulation of these objects, to make it easier to pass them with their
    associated dataloader properties to various functions, as well as providing
    various utility methods.

    The expected signature of the dataset, criterion, and networks are:

    .. code:
        for *inputs, labels in dataloader:
            out = network(*inputs)
            loss = criterion(out, labels)

    The expected signature to use :class:`LossFunction` is:

    .. code:
        loss_func = LossFunction(dataset, criterion, net)
        params = loss_func.initialize_parameters()
        loss = loss_func(params)
    '''

    def __init__(self, dataset: torch.utils.data.Dataset, criterion: Callable,
                 net: torch.nn.Module,  **dataloader_kwargs):
        '''
        Args:
            dataset (:class:`torch.utils.data.Dataset`): The dataset.
            criterion (callable): The criterion.
            net (:class:`torch.nn.Module`): The network architecture.

        In addition, keyword arguments such as num_workers and batch_size may
        be passed that will be used in instantiating the
        :class:`torch.utils.data.DataLoader`.
        '''
        super().__init__()

        self._dataloader = torch.utils.data.DataLoader(
            distributed_dataset_wrapper(dataset),
            **dataloader_kwargs,
        )
        # We need the criterion to *not* reduce the loss. For PyTorch's
        # built-in criteria, this should handle it.
        if isinstance(criterion, torch.nn.modules.loss._Loss):
            criterion.reduction = 'none'
        self._criterion = criterion

        self._net = net.to('meta')

    def criterion(self, out: torch.Tensor, labels: torch.Tensor) \
            -> torch.Tensor:
        loss = self._criterion(out, labels)
        if loss.shape != labels.shape:
            raise RuntimeError(
                f'Criterion should not reduce the loss. Got loss of shape '
                f'{loss.shape}, when expecting shape {labels.shape}.'
            )
        if loss.ndim > 1:
            loss = loss.flatten(1).sum(1)
        return loss

    def dataloader(self, device):
        for *inputs, labels in self._dataloader:
            inputs = tuple(x.to(device) for x in inputs)
            labels = labels.to(device)
            yield *inputs, labels

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        n_data = loss = 0
        param_dict = self._to_dict(params)

        for *inputs, labels in self.dataloader(params.device):
            # *-packing causes inputs to be a list, but functional_call
            # requires it to be a tuple
            out = torch.func.functional_call(
                self._net, param_dict, tuple(inputs)
            )
            loss += self.criterion(out, labels).sum()
            n_data += labels.shape[0]

        loss = sum_across_processes_(loss)
        n_data = torch.tensor([n_data], device=params.device)
        n_data = sum_across_processes_(n_data)
        return loss / n_data

    def gradient(self, params: torch.Tensor) -> torch.Tensor:
        '''
        Calculates the gradient of the network, and returns both it and the
        value of the loss.
        '''
        grads = torch.zeros_like(params)
        n_data = loss = 0

        for *inputs, labels in self.dataloader(params.device):
            # Calculate gradients. Note we sum instead of taking the mean, then
            # divide by the size of the dataset at the end, in case different
            # processes instead up with different numbers of data items.
            def compute_loss(params):
                param_dict = self._to_dict(params)
                # *-packing causes inputs to be a list, but functional_call
                # requires it to be a tuple
                out = torch.func.functional_call(
                    self._net, param_dict, tuple(inputs)
                )
                return self.criterion(out, labels).sum()
            compute_grad = torch.func.grad_and_value(compute_loss)
            batch_grads, batch_loss = compute_grad(params)
            grads += batch_grads
            n_data += labels.shape[0]
            loss += batch_loss

        # Note we do these operations in-place, so that grad_dict still
        # contains views of grad_tensor.
        n_data = torch.tensor(n_data, device=params.device)
        n_data = sum_across_processes_(n_data)
        grads = sum_across_processes_(grads)
        loss = sum_across_processes_(loss)
        grads /= n_data
        loss /= n_data

        return loss, grads

    def initialize_parameters(self, gain: float = sqrt(2.),
                              device: torch.device = torch.device('cpu'),
                              dtype: torch.dtype = torch.float32) \
            -> torch.Tensor:
        num_el = sum(p.numel() for p in self._net.parameters())
        out = torch.empty((num_el,), device=device, dtype=dtype)
        for n, p in self._to_dict(out).items():
            if n.endswith('weight'):
                torch.nn.init.xavier_normal_(p, gain=gain)
            else:
                if not n.endswith('bias'):
                    warnings.warn(
                        f'LossFunction does not know how to initialize '
                        f'parameter {n}; leaving as zeros.'
                    )
                torch.nn.init.zeros_(p)
        return out

    def _to_dict(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        This is a convenience method that takes as input a 1-dimensional
        :class:`torch.Tensor` and converts it into a dictionary of views on the
        input tensor suitable for being used as a dictionary of parameters.
        '''
        names, meta_params = zip(*self._net.named_parameters())
        shapes = [p.shape for p in meta_params]
        idxs = np.cumsum([prod(s) for s in shapes]).tolist()[:-1]
        param_list = params.tensor_split(idxs, 0)
        return {n: p.view(*s) for n, p, s in zip(names, param_list, shapes)}
