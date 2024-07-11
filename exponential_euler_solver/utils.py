from math import prod
from typing import Dict

import numpy as np
import torch


def _assert_mlp(net: torch.nn.Module):
    '''
    Checks if a network is a multi-layer perceptron of the form we can ingest.
    '''
    if not isinstance(net, torch.nn.Sequential):
        raise TypeError(
            'This function requires the module to be a torch.nn.Sequential'
        )
    for m in net:
        # We use the presence of parameters to distinguish between affine
        # layers and activation functions
        if len(list(m.parameters())) and not isinstance(m, torch.nn.Linear):
            raise TypeError(
                f'This function requires the module to be a multi-layer '
                f'perceptron; found a submodule that is a {type(m)}'
            )


def _params_tensor_to_dict(params: torch.Tensor, net: torch.nn.Module) \
        -> Dict[str, torch.Tensor]:
    '''
    This is a convenience method that takes as input a 1-dimensional
    :class:`torch.Tensor` and converts it into a dictionary of views on the
    input tensor suitable for being used as a dictionary of parameters.
    '''
    names, meta_params = zip(*net.named_parameters())
    shapes = [p.shape for p in meta_params]
    idxs = np.cumsum([prod(s) for s in shapes]).tolist()[:-1]
    param_list = params.tensor_split(idxs, 0)
    return {n: p.view(*s) for n, p, s in zip(names, param_list, shapes)}
