from typing import Callable, Tuple

import torch

__all__ = ['mhp']


def mhp(func: Callable, inputs: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    '''
    Modifies the :func:`torch.autograd.functional.vhp` to perform a matrix-
    Hessian product instead of vector-Hessian product.
    '''
    if m.shape[-1] == 1:
        # If dim == 1, then there's no need to involve the extra vmap apparatus
        # because we can treat it as a vector.
        _, out = torch.autograd.functional.vhp(
            lambda x: func(x.unsqueeze(-1)).squeeze(-1), inputs, m.squeeze(-1)
        )
        return out.unsqueeze(-1)

    else:
        # Ordinarily, _grad_preprocess would be called inside
        # torch.autograd.functional.vhp. However, this involves calling
        # retain_grad_ on each tensor, which torch.vmap does not like. We
        # therefore move that call outside.
        inputs = torch.autograd.functional._grad_preprocess(
            (inputs,), create_graph=False, need_graph=True
        )
        m = torch.autograd.functional._grad_preprocess(
            (m,), create_graph=False, need_graph=False
        )

        # We create this function so as to have a function of the appropriate
        # signature for torch.vmap.
        def _vhp(v: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
            return vhp(func, inputs, v)

        return torch.vmap(_vhp, -1, -1)(m)[0]


def vhp(func, inputs, v):
    # As discussed above, we replace torch.autograd.functional.vhp with our own
    # version that does not include _grad_preprocess, which we instead handle
    # outside of the function call, because _grad_preprocess includes
    # repeatedly calling retain_grad_, which torch.vmap does not like.
    with torch.enable_grad():
        outputs = func(*inputs)
        is_outputs_tuple, outputs = torch.autograd.functional._as_tuple(
            outputs, "outputs of the user-provided function", "vhp"
        )

        if is_outputs_tuple or not isinstance(outputs[0], torch.Tensor):
            raise RuntimeError(
                'The function given to vhp should return a single Tensor'
            )

        if outputs[0].nelement() != 1:
            raise RuntimeError(
                'The Tensor returned by the function given to vhp should '
                'contain a single element'
            )

        jac = torch.autograd.functional._autograd_grad(
            outputs, inputs, create_graph=True
        )

    grad_res = torch.autograd.functional._autograd_grad(
        jac, inputs, v, create_graph=False
    )
    vhp = torch.autograd.functional._fill_in_zeros(
        grad_res, inputs, False, False, 'double_back'
    )
    return vhp
