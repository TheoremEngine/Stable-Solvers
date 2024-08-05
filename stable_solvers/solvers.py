from dataclasses import dataclass
from typing import Optional
import warnings

import torch

from .eigen import loss_hessian_eigenvector
from .loss_functions import LossFunction

__all__ = [
    'AdaptiveGradientDescent', 'AdaptiveGradientDescentReport',
    'ExponentialEulerSolver', 'ExponentialEulerSolverReport',
    'GradientDescent', 'GradientDescentReport', 'Solver', 'SolverReport'
]


@dataclass
class SolverReport:
    dt: float
    loss: float


class Solver:
    '''
    This is a superclass for all available solvers. The expected use for this
    class is:

    .. code::
        loss_func = LossFunction(dataset=dataset, criterion=criterion)
        solver = Solver(net=net, loss=loss_func)
        ...
        for _ in range(num_iters):
            # Do analysis
            ...
            loss = solver.step()
    '''
    def __init__(self, params: torch.Tensor, loss: LossFunction,
                 error_if_unstable: bool = False):
        '''
        Args:
            params (:class:`torch.Tensor`): The parameters of the network that
            are being optimized.

            loss (:class:`LossFunction`): The loss function.
        '''
        self.params = params
        self.loss = loss
        self.error_if_unstable = error_if_unstable
        self.last_loss = float('inf')

    def _check_loss(self, loss: torch.Tensor):
        if self.last_loss < loss:
            if self.error_if_unstable:
                raise RuntimeError(
                    'Loss has increased, indicating training is unstable'
                )
            else:
                warnings.warn(
                    'Loss has increased, indicating training is unstable'
                )
        self.last_loss = loss

    def device(self) -> torch.device:
        return self.params.device

    def step(self) -> SolverReport:
        raise NotImplementedError()


@dataclass
class GradientDescentReport(SolverReport):
    gradient: Optional[torch.Tensor] = None


class GradientDescent(Solver):
    '''
    Performs conventional gradient descent without momentum, following the
    update rule:

    .. math::
        \\theta_{u+1} = \\theta_u -
        \\eta \\nabla_\\theta \\widetilde{\\mathcal{L}}(\\theta)

    Where :math:`\\theta_u` is the parameters at iteration :math:`u`,
    :math:`\\eta` is the learning rate, and
    :math:`\\widetilde{\\mathcal{L}}(\\theta)` is the training loss.

    This class is primarily provided for comparison purposes, but training with
    conventional gradient descent can be stable if the learning rate is small
    enough.
    '''
    def __init__(self, params: torch.Tensor, loss: LossFunction,
                 lr: float, report_gradient: bool = False,
                 error_if_unstable: bool = False):
        '''
        Args:
            params (:class:`torch.Tensor`): The parameters of the network that
            are being optimized.

            loss (:class:`LossFunction`): The loss function.

            lr (float): Learning rate.

            report_gradient (bool): Whether to include the gradient in the
            report returned at each step.

            error_if_unstable (bool): Whether to raise a :class:`RuntimeError`
            if training becomes unstable, as determined by the loss rising.
        '''
        super().__init__(
            params=params, loss=loss, error_if_unstable=error_if_unstable,
        )
        self.lr = lr
        self.report_gradient = report_gradient

    def step(self) -> SolverReport:
        loss, grads = self.loss.gradient(self.params)
        self._check_loss(loss)
        self.params -= self.lr * grads

        return GradientDescentReport(
            loss=loss.item(),
            dt=self.lr,
            gradient=(grads if self.report_gradient else None),
        )


@dataclass
class AdaptiveGradientDescentReport(SolverReport):
    sharpness: float
    gradient: Optional[torch.Tensor] = None
    eigvec: Optional[torch.Tensor] = None


class AdaptiveGradientDescent(Solver):
    '''
    Performs gradient descent without momentum, adapting the learning rate at
    every step to prevent entering the edge of stability:

    .. math::
        \\theta_{u+1} = \\theta_u -
        \\eta_u \\nabla_\\theta \\widetilde{\\mathcal{L}}(\\theta)

        \\eta_u = \\min\\left(\\eta_{\\max}, \\frac{1}
        {\\lambda^1(\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta_u))}
        \\right)

    Where :math:`\\theta_u` is the parameters at iteration :math:`u`,
    :math:`\\widetilde{\\mathcal{L}}(\\theta)` is the training loss,
    :math:`\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta)` is the
    Hessian matrix of the training loss, :math:`\\lambda^1(\\cdot)` is the top
    eigenvalue, and :math:`\\eta_{\\max}` is a hyperparameter.
    '''
    def __init__(self, params: torch.Tensor, loss: LossFunction,
                 lr: float, warmup_iters: int = 0, warmup_factor: float = 1.,
                 report_gradient: bool = False, report_eigvec: bool = False,
                 error_if_unstable: bool = False):
        '''
        Args:
            params (:class:`torch.Tensor`): The parameters of the network that
            are being optimized.

            loss (:class:`LossFunction`): The loss function.

            lr (float): Maximum learning rate. If the adaptive learning rate
            exceeds this value, it is truncated to be no higher than this.

            warmup_iters (int): If set, the maximum learning rate is initially
            set to a lower value for this many iterations, to damp out
            initial transients.

            warmup_factor (float): If set, the maximum learning rate is
            initially reduced by this factor, to damp out initial
            transients.

            report_gradient (bool): Whether to include the gradient in the
            report returned at each step.

            report_eigvec (bool): Whether to include the top eigenvector in the
            report returned at each step.

            error_if_unstable (bool): Whether to raise a :class:`RuntimeError`
            if training becomes unstable, as determined by the loss rising.
        '''
        super().__init__(
            params=params, loss=loss, error_if_unstable=error_if_unstable,
        )
        self.lr = lr
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.report_gradient = report_gradient
        self.report_eigvec = report_eigvec
        self.eigvec = None
        self.i = 0

    def step(self) -> AdaptiveGradientDescentReport:
        '''
        Takes a single step, returning a report.
        '''
        # Calculate the gradient
        loss, grads = self.loss.gradient(self.params)
        self._check_loss(loss)
        # Calculate the step size
        sharpness, self.eigvec = loss_hessian_eigenvector(
            self.loss, self.params, 1,
            init_eigvecs=self.eigvec, max_iters=1000,
        )
        scale = self.warmup_factor if (self.i < self.warmup_iters) else 1.
        step_size = min(scale * self.lr, 1. / sharpness.abs().item())
        # Perform update
        self.params -= step_size * grads
        self.i += 1

        return AdaptiveGradientDescentReport(
            loss=loss.item(),
            dt=step_size,
            sharpness=sharpness.item(),
            gradient=(grads if self.report_gradient else None),
            eigvec=(self.eigvec if self.report_eigvec else None),
        )


@dataclass
class ExponentialEulerSolverReport(SolverReport):
    eigvals: torch.Tensor
    gradient: Optional[torch.Tensor] = None
    eigvecs: Optional[torch.Tensor] = None


class ExponentialEulerSolver(Solver):
    '''
    Uses the exponential Euler method from `Lowell and Kastner 2024
    <https://arxiv.org/abs/2406.00127>`_:

    .. math::
        \\theta_{u+1} = \\theta_u -
        \\sum_{m=1}^k c_u^m r_u^m
        v^m(\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta)) -
        \\eta_u w_u

        r_u^m =
        \\min\\left(\\frac{1}
        {\\lambda^m(\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta))}
        \\left(e^{
        -\\lambda^m(\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta)) t
        } - 1\\right), \\eta_{\\max}\\right)

        c_u^m = \\nabla_\\theta \\widetilde{\\mathcal{L}}(\\theta_u)
        \\cdot v^m(\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta))

        w_u = \\nabla_\\theta \\widetilde{\\mathcal{L}}(\\theta_u) -
        \\sum_{m=1}^k c_u^m
        v^m(\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta))

        \\eta_u = \\max\\left(\\frac{1}{
        \\lambda^{k+1}(\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta))
        }, \\eta_{\\max}\\right)

    Where :math:`\\theta_u` is the parameters at iteration :math:`u`,
    :math:`\\widetilde{\\mathcal{L}}(\\theta)` is the training loss,
    :math:`\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta)` is the
    Hessian matrix of the training loss, :math:`\\lambda^m(\\cdot)` is the
    :math:`m` th top eigenvalue, :math:`v^m(\\cdot)` is the :math:`m` th top
    eigenvector, :math:`k` is the expected dimension of the stiff subspace, and
    :math:`\\eta_{\\max}` is a hyperparameter.

    Essentially, where conventional gradient descent approximates the loss
    function as a locally linear function, the exponential Euler solver
    approximates it as a locally quadratic function. Since recovering all of
    the quadratic terms in the Taylor expansion requires computing all of the
    eigenvalues and eigenvectors of
    :math:`\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta)`, which is
    intractable, it instead only captures the largest and most influential
    quadratic components, and settles for a linear approximation in the other
    directions. It additionally calculates a learning rate by using one more
    eigenvalue, analogous to the :class:`AdaptiveGradientDescent` solver, and
    flows along the gradient flow by a time step equal to that learning rate.

    The stiff dimension should be set to the dimension of the highly curved
    subspace. If the stiff dimension is set too high, the solver will be slow
    because it will be calculating more eigenvectors than it needs to. If it is
    set too low, it will be slow because it will not be fully compensating for
    the curvature of the loss landscape. Fortunately, in practice, the
    dimension of the highly curved subspace is equal to the dimension of the
    network outputs, reduced by one if using the cross-entropy loss. For
    example, a classifier trained using cross-entropy loss on a dataset with 10
    classes would have a stiff dimension of 9. A regression network trained
    using mean-squared error to predict a single value would have a stiff
    dimension of 1.
    '''
    def __init__(self, params: torch.Tensor, loss: LossFunction,
                 max_step_size: float, stiff_dim: int, warmup_iters: int = 0,
                 warmup_factor: float = 1., report_gradient: bool = False,
                 report_eigvecs: bool = False,
                 error_if_unstable: bool = False):
        '''
        Args:
            params (:class:`torch.Tensor`): The parameters of the network that
            are being optimized.

            loss (:class:`LossFunction`): The loss function.

            max_step_size (float): Maximum step size.

            stiff_dim (int): Dimension of the expected "stiff" component
            of the loss landscape, generally equal to the number of
            network outputs.

            warmup_iters (int): If set, the maximum step size is initially
            set to a lower value for this many iterations, to damp out
            initial transients.

            warmup_factor (float): If set, the maximum step size is
            initially reduced by this factor, to damp out initial
            transients.

            report_gradient (bool): Whether to include the gradient in the
            report returned at each step.

            report_eigvecs (bool): Whether to include the top eigenvectors in
            the report returned at each step.

            error_if_unstable (bool): Whether to raise a :class:`RuntimeError`
            if training becomes unstable, as determined by the loss rising.
        '''
        super().__init__(
            params=params, loss=loss, error_if_unstable=error_if_unstable,
        )
        self.max_step_size = max_step_size
        self.stiff_dim = stiff_dim
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.report_gradient = report_gradient
        self.report_eigvecs = report_eigvecs
        self.eigvecs = None
        self.i = 0

    def step(self) -> ExponentialEulerSolverReport:
        '''
        Takes a single step, returning a report.
        '''
        # Calculate the gradient
        loss, grads = self.loss.gradient(self.params)
        self._check_loss(loss)
        # Calculate the eigenvectors. Note that we add one to the stiff
        # dimension so we can also calculate the step size adaptively for the
        # non-stiff component.
        eigvals, self.eigvecs = loss_hessian_eigenvector(
            self.loss, self.params, self.stiff_dim + 1,
            init_eigvecs=self.eigvecs, max_iters=1000,
        )
        # Break the gradient into the components lying on the stiff
        # eigenvectors and the bulk remainder.
        stiff_projections = grads @ self.eigvecs[:, :-1]
        bulk = grads - self.eigvecs[:, :-1] @ stiff_projections
        # Calculate bulk component of the step
        scale = self.warmup_factor if (self.i < self.warmup_iters) else 1.
        step_size = min(
            scale * self.max_step_size, 1. / eigvals[-1].abs().item()
        )
        step = bulk * -step_size
        # Calculate stiff component of the step
        stiff_step = (1 - (-eigvals[:-1] * step_size).exp()) / eigvals[:-1]
        stiff_step = stiff_step.clamp_(max=step_size)
        step -= self.eigvecs[:, :-1] @ (stiff_step * stiff_projections)
        self.params += step
        self.i += 1

        return ExponentialEulerSolverReport(
            loss=loss.item(),
            eigvals=eigvals,
            dt=step_size,
            gradient=(grads if self.report_gradient else None),
            eigvecs=(self.eigvecs if self.report_eigvecs else None),
        )
