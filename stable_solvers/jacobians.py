from typing import Callable, Dict, Tuple

import torch

from .utils import _assert_mlp, _params_tensor_to_dict

__all__ = ['make_layerwise_jacobian_function']


def make_layerwise_jacobian_function(net: torch.nn.Sequential,
                                     params: torch.Tensor) -> Callable:
    '''
    Returns a function that takes a set of inputs and returns the layerwise
    Jacobian matrices:

    .. code-block::
        jac_func = make_layerwise_jacobian_function(net, params)
        jac_0, jac_1, ... = jac_func(xs)

    The network must be a multi-layer perceptron instantiated as a
    :class:`torch.nn.Sequential`.

    Args:
        net (:class:`torch.nn.Sequential`): The network to use.

        params (:class:`torch.Tensor`): The network parameters.
    '''
    _assert_mlp(net)
    param_dict = _params_tensor_to_dict(params, net)

    affine_idxs = [
        i for i, m in enumerate(net) if isinstance(m, torch.nn.Linear)
    ]
    preact_idxs = [0] + [(i + 1) for i in affine_idxs]
    subnets = [
        net[i_0:i_1]
        for i_0, i_1 in zip(preact_idxs[:-1], preact_idxs[1:])
    ]
    subnet_params = [
        {f'{i}.{k}': param_dict[f'{i}.{k}'] for k in ['weight', 'bias']}
        for i in affine_idxs
    ]

    # _aux_wrapper wraps the subnet so it returns a pair, containing the output
    # twice. We do this so we can get the preactivations as well as the
    # Jacobian, following the docstring for torch.func.jacrev.
    jac_funcs = [
        torch.func.jacrev(_aux_wrapper(subnet, p), has_aux=True)
        for subnet, p in zip(subnets, subnet_params)
    ]

    def func(x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out, x = [], x.unsqueeze(0)
        for jac_func in jac_funcs:
            jac, x = jac_func(x)
            out.append(jac.squeeze(0).squeeze(1).T)
        return tuple(out)

    return torch.func.vmap(func)


def _aux_wrapper(net: Callable, param_dict: Dict[str, torch.Tensor]) \
        -> Callable:
    '''
    Modifies a function so that it returns a pair consisting of the output
    twice.
    '''
    def func(x):
        out = torch.func.functional_call(net, param_dict, x)
        return out, out
    return func
