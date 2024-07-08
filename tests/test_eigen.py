import unittest

import numpy as np
import torch
import torch.linalg as la

import exponential_euler_solver as euler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class EigvecsTest(unittest.TestCase):
    def test_g_eigvecs(self):
        xs = torch.randn((4, 4), device=DEVICE)
        ys = torch.zeros((4, 2), device=DEVICE)
        ys[0, 0] = ys[1, 0] = ys[2, 1] = ys[3, 1] = 1.
        ds = torch.utils.data.TensorDataset(xs, ys)
        # Use MSELoss so the Hessian of the criterion is constant
        criterion = torch.nn.MSELoss()

        net = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(True),
            torch.nn.Linear(16, 2)
        )
        net = net.to(DEVICE)

        names, param_list = zip(*net.named_parameters())
        idxs = np.cumsum([0] + [p.numel() for p in param_list])
        shapes = [p.shape for p in param_list]
        param_tensor = torch.cat([p.flatten() for p in param_list])

        def compute_out(params):
            params = {
                n: params[i_0:i_1].view(*s)
                for n, i_0, i_1, s in zip(names, idxs[:-1], idxs[1:], shapes)
            }
            return torch.func.functional_call(net, params, xs)

        # jacs will be of shape (N, dim Z, dim P).
        jacs = torch.autograd.functional.jacobian(compute_out, param_tensor)
        g = 2 * torch.bmm(jacs.permute(0, 2, 1), jacs).mean(0)
        eigvals_1, _ = la.eigh(g)
        # Sort by magnitude, because that's what our power_iteration_method
        # will return
        eigvals_1 = eigvals_1[torch.argsort(eigvals_1.abs(), descending=True)]

        loss_func = euler.LossFunction(
            dataset=ds, criterion=criterion, batch_size=4, net=net,
        )
        eigvals_2, _ = euler.g_eigenvector(
            loss_func, param_tensor, tol=1e-6, n=2, max_iters=1000
        )

        # Sometimes, there are multiple eigenvalues whose absolute values are
        # almost identical, but which have different signs, so we can't really
        # tell the correct order of them.
        self.assertTrue(
            (((eigvals_2 - eigvals_1[:2]).abs() < 1e-4) |
             ((eigvals_2 + eigvals_1[:2]).abs() < 1e-4)).all()
        )

        # We do not test the eigenvectors. We do this deliberately. If we have
        # two eigenvalues close to each other - which we very frequently do for
        # H - then the eigenvectors will be a more-or-less random sample of
        # orthogonal unit vectors from the space spanned by those eigenvalues'
        # corresponding eigenvectors, so they are harder to check than is worth
        # it.

    def test_h_eigvecs(self):
        xs = torch.randn((4, 4), device=DEVICE)
        ys = torch.zeros((4, 2), device=DEVICE)
        ys[0, 0] = ys[1, 0] = ys[2, 1] = ys[3, 1] = 1.
        ds = torch.utils.data.TensorDataset(xs, ys)
        # Use MSELoss so the Hessian of the criterion is constant
        criterion = torch.nn.MSELoss()

        net = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(True),
            torch.nn.Linear(16, 2)
        )
        net = net.to(DEVICE)

        names, param_list = zip(*net.named_parameters())
        idxs = np.cumsum([0] + [p.numel() for p in param_list])
        shapes = [p.shape for p in param_list]
        param_tensor = torch.cat([p.flatten() for p in param_list])

        def compute_out(params):
            params = {
                n: params[i_0:i_1].view(*s)
                for n, i_0, i_1, s in zip(names, idxs[:-1], idxs[1:], shapes)
            }
            z = torch.func.functional_call(net, params, xs)
            return 2 * ((z - ys).detach() * z).sum() / ys.shape[0]

        h = torch.func.hessian(compute_out)(param_tensor)
        eigvals_1, eigvecs_1 = la.eigh(h)
        # Sort by magnitude, because that's what our power_iteration_method
        # will return
        idxs = torch.argsort(eigvals_1.abs(), descending=True)
        eigvals_1, _ = eigvals_1[idxs], eigvecs_1[idxs]

        loss_func = euler.LossFunction(
            dataset=ds, criterion=criterion, batch_size=4, net=net,
        )
        eigvals_2, _ = euler.h_eigenvector(
            loss_func, param_tensor, tol=1e-6, n=2, max_iters=1000
        )

        # Sometimes, there are multiple eigenvalues whose absolute values are
        # almost identical, but which have different signs, so we can't really
        # tell the correct order of them.
        self.assertTrue(
            (((eigvals_2 - eigvals_1[:2]).abs() < 1e-4) |
             ((eigvals_2 + eigvals_1[:2]).abs() < 1e-4)).all()
        )

        # We do not test the eigenvectors. We do this deliberately. If we have
        # two eigenvalues close to each other - which we very frequently do for
        # H - then the eigenvectors will be a more-or-less random sample of
        # orthogonal unit vectors from the space spanned by those eigenvalues'
        # corresponding eigenvectors, so they are harder to check than is worth
        # it.

    def test_loss_hessian_eigvecs(self):
        xs = torch.randn((4, 4), device=DEVICE)
        ys = torch.tensor([0, 0, 1, 1], device=DEVICE)
        ds = torch.utils.data.TensorDataset(xs, ys)
        criterion = torch.nn.CrossEntropyLoss()

        net = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.ReLU(True),
            torch.nn.Linear(16, 2)
        )
        net = net.to(DEVICE)

        names, param_list = zip(*net.named_parameters())
        idxs = np.cumsum([0] + [p.numel() for p in param_list])
        shapes = [p.shape for p in param_list]
        param_tensor = torch.cat([p.flatten() for p in param_list])

        def compute_loss(params):
            params = {
                n: params[i_0:i_1].view(*s)
                for n, i_0, i_1, s in zip(names, idxs[:-1], idxs[1:], shapes)
            }
            zs = torch.func.functional_call(net, params, xs)
            return criterion(zs, ys)

        hess = torch.func.hessian(compute_loss)(param_tensor)
        eigvals_1, eigvecs_1 = la.eigh(hess)
        eigvals_1 = torch.flip(eigvals_1, dims=(0,))
        eigvecs_1 = torch.flip(eigvecs_1, dims=(1,))

        loss_func = euler.LossFunction(
            dataset=ds, criterion=criterion, batch_size=4, net=net,
        )
        eigvals_2, eigvecs_2 = euler.loss_hessian_eigenvector(
            loss_func, param_tensor, tol=1e-6, n=2, max_iters=1000
        )

        self.assertTrue(((eigvals_2 - eigvals_1[:2]).abs() < 1e-4).all())

        # We do not test the eigenvectors. We do this deliberately. If we have
        # two eigenvalues close to each other - which we very frequently do for
        # H - then the eigenvectors will be a more-or-less random sample of
        # orthogonal unit vectors from the space spanned by those eigenvalues'
        # corresponding eigenvectors, so they are harder to check than is worth
        # it.


if __name__ == '__main__':
    unittest.main()
