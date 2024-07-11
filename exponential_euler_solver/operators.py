import torch

from .autograd import mhp
from .distributed import broadcast_across_processes_, sum_across_processes_
from .loss_functions import LossFunction
from .utils import _params_tensor_to_dict

__all__ = ['GMatrix', 'HMatrix', 'LossHessian', 'Operator']


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
        \\mathbb{E}_{x,y\\sim\\mathcal{T}} \\left(
        \\frac{\\partial^2 l(f(x, \\theta), y)}
        {\\partial \\theta_i \\partial \\theta_j})(x, y) \\right)

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
                z = torch.func.functional_call(
                    self.loss_func._net,
                    _params_tensor_to_dict(params, self.loss_func._net),
                    *inputs
                )
                return self.loss_func.criterion(z, labels).sum()

            out += mhp(compute_loss, self.params, m)
            n_data += labels.shape[0]

        out = sum_across_processes_(out)
        n_data = torch.tensor([n_data], device=self.device)
        n_data = sum_across_processes_(n_data)
        out /= n_data

        return out


class GMatrix(Operator):
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

        def criterion(z, y):
            return self.loss_func.criterion(z, y).sum()

        criterion_grad = torch.vmap(torch.func.grad(criterion))

        for *inputs, labels in self.loss_func.dataloader(self.device):
            # We can't use torch.func here because there's no
            # Hessian-vector product in torch.func yet.
            # We can't use torch.func here because there's no
            # Hessian-vector product in torch.func yet.
            def func(params: torch.Tensor) -> torch.Tensor:
                z = torch.func.functional_call(
                    self.loss_func._net,
                    _params_tensor_to_dict(params, self.loss_func._net),
                    *inputs
                )
                dz_dl = criterion_grad(z, labels)
                return (
                    self.loss_func.criterion(z, labels).sum() -
                    (dz_dl.detach() * z).sum()
                )

            out += mhp(func, self.params, m)
            n_data += labels.shape[0]

        out = sum_across_processes_(out)
        n_data = torch.tensor([n_data], device=self.device)
        n_data = sum_across_processes_(n_data)
        out /= n_data

        return out


class HMatrix(Operator):
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

        def criterion(z, y):
            return self.loss_func.criterion(z, y).sum()

        criterion_grad = torch.vmap(torch.func.grad(criterion))

        for *inputs, labels in self.loss_func.dataloader(self.device):
            # We can't use torch.func here because there's no
            # Hessian-vector product in torch.func yet.
            def func(params: torch.Tensor) -> torch.Tensor:
                z = torch.func.functional_call(
                    self.loss_func._net,
                    _params_tensor_to_dict(params, self.loss_func._net),
                    *inputs
                )
                dz_dl = criterion_grad(z, labels)
                return (dz_dl.detach() * z).sum()

            out += mhp(func, self.params, m)
            n_data += labels.shape[0]

        out = sum_across_processes_(out)
        n_data = torch.tensor([n_data], device=self.device)
        n_data = sum_across_processes_(n_data)
        out /= n_data

        return out
