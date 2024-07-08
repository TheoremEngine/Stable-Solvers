from typing import Optional, Tuple

import torch
import torch.linalg as la

from .loss_functions import LossFunction
from .operators import GMatrix, HMatrix, LossHessian, Operator


__all__ = [
    'g_eigenvector', 'h_eigenvector', 'loss_hessian_eigenvector',
    'power_iteration_method'
]


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
    # Construct the initial set of eigenvectors
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
        vectors, r = la.qr(vectors, mode='reduced')
        new_eigvals = torch.diagonal(r)

        if (eigvals - new_eigvals).norm(p=1) < tol:
            break

        eigvals = new_eigvals

        vectors = op(vectors)

    else:
        raise RuntimeError(
            f'power_iteration_method exceeded maximum number of '
            f'iterations {max_iters}'
        )

    return new_eigvals, vectors


def g_eigenvector(loss: LossFunction, params: torch.Tensor,
                  n: int = 1, max_iters: int = 100, tol: float = 1e-4,
                  init_eigvecs: Optional[torch.Tensor] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Calculates the top eigenvalue and eigenvector of the $G$ matrix of the loss
    function using the power iteration method. The $G$ matrix is defined by:

    .. math::
        G_{i,j} = \\mathbb{E}_{x,y\\sim\\mathcal{T} \\left(\\sum_{q,r}
        \\frac{\\partial^2 l}{\\partial z_q \\partial z_r}
        \\frac{\\partial f_q}{\\partial \\theta_i}
        \\frac{\\partial f_r}{\\partial \\theta_j} \\right)

    Where $\\mathcal{T}$ is the training set, $l$ is the criterion, and
    $f(x, \\theta)$ is the output of the neural network with inputs $x$
    and parameters $\\theta$.

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
    op = GMatrix(loss, params)
    return power_iteration_method(
        op, n=n, max_iters=max_iters, tol=tol, init_eigvecs=init_eigvecs
    )


def h_eigenvector(loss: LossFunction, params: torch.Tensor,
                  n: int = 1, max_iters: int = 100, tol: float = 1e-4,
                  init_eigvecs: Optional[torch.Tensor] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Calculates the top eigenvalue and eigenvector of the $H$ matrix of the loss
    function using the power iteration method. The $H$ matrix is defined by:

    .. math::
        H_{i,j} = \\mathbb{E}_{x,y\\sim\\mathcal{T} \\left(\\sum_q
        \\frac{\\partial l}{\\partial z_q}
        \\frac{\\partial^2 f_q}{\\partial \\theta_i \\theta_j} \\right)

    Where $\\mathcal{T}$ is the training set, $l$ is the criterion, and
    $f(x, \\theta)$ is the output of the neural network with inputs $x$
    and parameters $\\theta$.

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
    op = HMatrix(loss, params)
    return power_iteration_method(
        op, n=n, max_iters=max_iters, tol=tol, init_eigvecs=init_eigvecs
    )


def loss_hessian_eigenvector(loss: LossFunction, params: torch.Tensor,
                             n: int = 1, max_iters: int = 100,
                             tol: float = 1e-4,
                             init_eigvecs: Optional[torch.Tensor] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Calculates the top eigenvalue and eigenvector of the Hessian matrix of the
    loss function using the power iteration method. The Hessian matrix of the
    loss function is defined by:

    .. math::
        (\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta))_{i,j} =
        \\mathbb{E}_{x,y\\sim\\mathcal{T}} \\left(
        \\frac{\\partial^2 l(f(x, \\theta), y)}
        {\\partial \\theta_i \\partial \\theta_j})(x, y) \\right)

    Where $\\mathcal{T}$ is the training set, $l$ is the criterion, and
    $f(x, \\theta)$ is the output of the neural network with inputs $x$
    and parameters $\\theta$.

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
