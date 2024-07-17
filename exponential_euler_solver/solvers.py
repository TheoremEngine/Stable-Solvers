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
    every step to prevent entering the edge of stability.
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
        self.eigvec = None

    def step(self) -> AdaptiveGradientDescentReport:
        # Calculate the gradient
        loss, grads = self.loss.gradient(self.params)
        # Calculate the step size
        sharpness, self.eigvec = loss_hessian_eigenvector(
            self.loss, self.params, 1,
            init_eigvecs=self.eigvec, max_iters=1000,
        )
        step_size = min(self.lr, 1. / sharpness.item())
        # Perform update
        self.params -= step_size * grads
        return AdaptiveGradientDescentReport(
            loss=loss.item(), dt=step_size, sharpness=sharpness.item(),
        )


@dataclass
class ExponentialEulerSolverReport(SolverReport):
    eigvals: torch.Tensor


class ExponentialEulerSolver(Solver):
    def __init__(self, params: torch.Tensor, loss: LossFunction,
                 max_step_size: float, stiff_dim: int):
        '''
        Args:
            params (:class:`torch.Tensor`): The parameters of the network that
                are being optimized.
            loss (:class:`LossFunction`): The loss function.
            max_step_size (float): Maximum step size.
            stiff_dim (int): Dimension of the expected "stiff" component
                of the loss landscape, generally equal to the number of
                network outputs.
        '''
        super().__init__(params=params, loss=loss)
        self.max_step_size = max_step_size
        self.stiff_dim = stiff_dim
        self.eigvecs = None

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
        step_size = min(self.max_step_size, 1. / eigvals[-1].abs().item())
        step = bulk * -step_size
        # Calculate stiff component of the step
        stiff_step = (1 - (-eigvals[:-1] * step_size).exp()) / eigvals[:-1]
        stiff_step = stiff_step.clamp_(max=step_size)
        step -= self.eigvecs[:, :-1] @ (stiff_step * stiff_projections)
        self.params += step

        return ExponentialEulerSolverReport(
            loss=loss.item(), eigvals=eigvals, dt=step_size,
        )
