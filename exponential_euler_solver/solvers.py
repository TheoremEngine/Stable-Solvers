from dataclasses import dataclass
import torch

from .eigen import loss_hessian_eigenvector
from .loss_functions import LossFunction

__all__ = [
    'AdaptiveGradientDescent', 'AdaptiveGradientDescentReport',
    'ExponentialEulerSolver', 'ExponentialEulerSolverReport',
    'GradientDescent', 'Solver', 'SolverReport'
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
    def __init__(self, params: torch.Tensor, loss: LossFunction):
        '''
        Args:
            params (:class:`torch.Tensor`): The parameters of the network that
                are being optimized.
            loss (:class:`LossFunction`): The loss function.
        '''
        self.params = params
        self.loss = loss

    def device(self) -> torch.device:
        return self.params.device

    def step(self) -> SolverReport:
        raise NotImplementedError()


class GradientDescent(Solver):
    '''
    Performs conventional gradient descent without momentum.
    '''
    def __init__(self, params: torch.Tensor, loss: LossFunction,
                 lr: float):
        '''
        Args:
            params (:class:`torch.Tensor`): The parameters of the network that
                are being optimized.
            loss (:class:`LossFunction`): The loss function.
            lr (float): Learning rate.
        '''
        super().__init__(params=params, loss=loss)
        self.lr = lr

    def step(self) -> SolverReport:
        loss, grads = self.loss.gradient(self.params)
        self.params -= self.lr * grads
        return SolverReport(loss=loss.item(), dt=self.lr)


@dataclass
class AdaptiveGradientDescentReport(SolverReport):
    sharpness: float


class AdaptiveGradientDescent(Solver):
    '''
    Performs gradient descent without momentum, adapting the learning rate at
    every step to prevent entering the edge of stability:

    ..math::
        \\theta_{u+1} = \\theta_u -
        \\eta_u \\nabla_\\theta \\widetilde{\\mathcal{L}}(\\theta)

        \\eta_u = \\min\\left(\\eta_{\\max}, \\frac{1}
        {\\lambda^1(\\mathcal{H}_\\theta \\widetilde{\\mathcal{L}}(\\theta_u))}
        \\right)
    '''
    def __init__(self, params: torch.Tensor, loss: LossFunction,
                 lr: float, warmup_iters: int = 0, warmup_factor: float = 1.):
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
        '''
        super().__init__(params=params, loss=loss)
        self.lr = lr
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.eigvec = None
        self.i = 0

    def step(self) -> AdaptiveGradientDescentReport:
        # Calculate the gradient
        loss, grads = self.loss.gradient(self.params)
        # Calculate the step size
        sharpness, self.eigvec = loss_hessian_eigenvector(
            self.loss, self.params, 1,
            init_eigvecs=self.eigvec, max_iters=1000,
        )
        scale = self.warmup_factor if (self.i < self.warmup_iters) else 1.
        step_size = min(scale * self.lr, 1. / sharpness.item())
        # Perform update
        self.params -= step_size * grads
        self.i += 1
        return AdaptiveGradientDescentReport(
            loss=loss.item(), dt=step_size, sharpness=sharpness.item(),
        )


@dataclass
class ExponentialEulerSolverReport(SolverReport):
    eigvals: torch.Tensor


class ExponentialEulerSolver(Solver):
    def __init__(self, params: torch.Tensor, loss: LossFunction,
                 max_step_size: float, stiff_dim: int, warmup_iters: int = 0,
                 warmup_factor: float = 1.):
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
        '''
        super().__init__(params=params, loss=loss)
        self.max_step_size = max_step_size
        self.stiff_dim = stiff_dim
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.eigvecs = None
        self.i = 0

    def step(self) -> ExponentialEulerSolverReport:
        # Calculate the gradient
        loss, grads = self.loss.gradient(self.params)
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
            loss=loss.item(), eigvals=eigvals, dt=step_size,
        )
