import unittest

import numpy as np
import torch
import torch.linalg as la

import exponential_euler_solver as euler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LossHessianTest(unittest.TestCase):
    def test_eigvecs(self):
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

        loss_func = euler.LossFunction(
            dataset=ds, criterion=criterion, batch_size=4, net=net,
        )
        eigvals_2, eigvecs_2 = euler.loss_hessian_eigenvector(
            loss_func, param_tensor, tol=1e-6, n=2, max_iters=1000
        )

        self.assertTrue((eigvals_2[0] - eigvals_1[-1]).abs() < 1e-4)
        self.assertTrue((eigvals_2[1] - eigvals_1[-2]).abs() < 1e-4)
        self.assertTrue(
            ((eigvecs_2[:, 0] - eigvecs_1[:, -1]) < 0.01).all() or
            ((eigvecs_2[:, 0] + eigvecs_1[:, -1]) < 0.01).all(),
        )
        self.assertTrue(
            ((eigvecs_2[:, 1] - eigvecs_1[:, -2]) < 0.01).all() or
            ((eigvecs_2[:, 1] + eigvecs_1[:, -2]) < 0.01).all(),
        )


if __name__ == '__main__':
    unittest.main()
