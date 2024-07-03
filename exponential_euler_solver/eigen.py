from typing import Optional, Tuple

import torch
import torch.linalg as la

from .autograd import mhp
from .distributed import broadcast_across_processes_, sum_across_processes_
from .loss_functions import LossFunction


__all__ = [
    'LossHessian', 'loss_hessian_eigenvector', 'Operator',
    'power_iteration_method'
]


class Operator(torch.nn.Module):
    '''
    This is an abstract base class for a linear operator on a vector. A method
    and a property are required: :meth:`forward`, which corresponds to the
    operator's operation, and :prop:`dim`, which must return the input
    dimension of the operator. This is used to encapsulate linear operations
    where the matrix is too large to be materialized in memory.
    '''
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dim(self) -> int:
        raise NotImplementedError()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def init(self, n_dim: int = 1) -> torch.Tensor:
        out = torch.randn(
            (self.dim, n_dim), device=self.device, dtype=self.dtype
        )
        return broadcast_across_processes_(out)


class LossHessian(Operator):
    '''
    This operator corresponds to the Hessian of the loss function:

    .. math::
        (\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta))_{i,j} =
        \\mathbb{E}_{x,y\\sim\\mathcal{T}}
        \\frac{\\partial^2 l(f(x, \\theta), y)}
        {\\partial \\theta_i \\partial \\theta_j})(x, y)

    Where $\\mathcal{T}$ is the training set, $l$ is the criterion, and
    $f(x, \\theta)$ is the output of the neural network with inputs $x$
    and parameters $\\theta$.
    '''
    def __init__(self, loss_func: LossFunction, params: torch.Tensor):
        '''
        Args:
            loss_func (:class:`LossFunction`): The loss function.
            params (:class:`torch.Tensor`): The network parameters.
        '''
        super().__init__()
        self.loss_func = loss_func
        if params.ndim != 1:
            raise ValueError(params.shape)
        self.params = params

    @property
    def device(self) -> torch.device:
        return self.params.device

    @property
    def dim(self) -> int:
        return self.params.shape[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.params.dtype

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        out = n_data = 0

        for *inputs, labels in self.loss_func.dataloader(self.device):
            # We can't use torch.func here because there's no
            # Hessian-vector product in torch.func yet.
            def compute_loss(params: torch.Tensor) -> torch.Tensor:
                x = torch.func.functional_call(
                    self.loss_func._net,
                    self.loss_func._to_dict(params),
                    *inputs
                )
                return self.loss_func.criterion(x, labels).sum()

            out += mhp(compute_loss, (self.params,), (m,))[0]
            n_data += labels.shape[0]

        out = sum_across_processes_(out)
        n_data = torch.tensor([n_data], device=self.device)
        n_data = sum_across_processes_(n_data)
        out /= n_data

        return out


def power_iteration_method(op: Operator, n: int = 1, max_iters: int = 100,
                           tol: float = 0.01,
                           init_eigvecs: Optional[torch.Tensor] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Calculates the top eigenvalue and eigenvector of an operator using the
    power iteration method. This method works by repeatedly calculating:

    .. math::
        v_{n + 1} = \\mbox{Op}(v_n) / || \\mbox{Op}(v_n) ||

    The process terminates once the difference between the iterations falls
    below a specified tolerance. The key advantage of this method is that it
    does not require the operator to be materialized in memory; only
    operator-vector products are needed.

    Args:
        op (:class:`Operator`): The operator.
        n (int): The number of eigenvalue-eigenvector pairs to calculate.
        max_iters (int): The maximum number of iterations to perform. If the
            procedure has not converged after this many iterations, a
            RuntimeError is raised.
        tol (float): The procedure declares convergence once the difference
            between iterations falls below this tolerance in the 1-norm.
        init_eigvecs (optional, :class:`torch.Tensor`): The initialization
            value. If not provided, this is randomly initialized. A sensible
            choice of initialization - for example, reusing the last
            eigenvector when repeatedly calculating the eigenvector during
            training - can radically reduce the number of iterations required.
    '''
    if init_eigvecs is None:
        vectors = op.init(n)
    elif not isinstance(init_eigvecs, torch.Tensor):
        raise TypeError(init_eigvecs)
    elif (init_eigvecs.shape != (op.dim, n)):
        raise RuntimeError(
            f'Shape mismatch in initialization for number of requested '
            f'eigenvalues: {init_eigvecs.shape} vs. {op.dim, n}'
        )
    else:
        vectors = init_eigvecs

    eigvals = torch.tensor([float('inf')] * n, device=op.device)
    for _ in range(max_iters):
        new_vectors = op(vectors)

        new_vectors, r = la.qr(new_vectors, mode='reduced')
        new_eigvals = torch.diagonal(r)

        if (eigvals - new_eigvals).norm(p=1) < tol:
            break

        vectors, eigvals = new_vectors, new_eigvals

    else:
        raise RuntimeError(
            f'power_iteration_method exceeded maximum number of '
            f'iterations {max_iters}'
        )

    return new_eigvals, new_vectors


def loss_hessian_eigenvector(loss: LossFunction, params: torch.Tensor,
                             n: int = 1, max_iters: int = 100,
                             tol: float = 1e-4,
                             init_eigvecs: Optional[torch.Tensor] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Calculates the top eigenvalue and eigenvector of the Hessian matrix of the
    loss function using the power iteration method.

    Args:
        loss (:class:`LossFunction`): The loss function.
        params (:class:`torch.Tensor`): The network parameters.
        n (int): The number of eigenvalue-eigenvector pairs to calculate.
        max_iters (int): The maximum number of iterations to perform. If the
            procedure has not converged after this many iterations, a
            RuntimeError is raised.
        tol (float): The procedure declares convergence once the difference
            between iterations falls below this tolerance in the 1-norm.
        init_eigvecs (optional, :class:`torch.Tensor`): The initialization
            value. If not provided, this is randomly initialized. A sensible
            choice of initialization - for example, reusing the last
            eigenvector when repeatedly calculating the eigenvector during
            training - can radically reduce the number of iterations required.
    '''
    op = LossHessian(loss, params)
    return power_iteration_method(
        op, n=n, max_iters=max_iters, tol=tol, init_eigvecs=init_eigvecs
    )
