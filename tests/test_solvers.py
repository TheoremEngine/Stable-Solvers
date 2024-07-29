import unittest

import torch

import stable_solvers as solvers
from stable_solvers.utils import _params_tensor_to_dict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SolverTest(unittest.TestCase):
    def test_gradient_descent(self):
        criterion = torch.nn.CrossEntropyLoss()
        xs = torch.randn((4, 3), device=DEVICE)
        ys = torch.tensor([0, 0, 1, 1], device=DEVICE, dtype=torch.int64)
        ds = torch.utils.data.TensorDataset(xs, ys)

        net_1 = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.ReLU(True),
            torch.nn.Linear(8, 2)
        )
        net_1 = net_1.to(DEVICE)
        net_2 = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.ReLU(True),
            torch.nn.Linear(8, 2)
        )
        net_2 = net_2.to(DEVICE)
        net_2.load_state_dict(net_1.state_dict())

        opt = torch.optim.SGD(net_1.parameters(), lr=0.1, momentum=0.)
        out = net_1(xs)
        loss_1 = criterion(out, ys).mean()
        loss_1.backward()
        opt.step()

        params = torch.cat([p.flatten() for p in net_2.parameters()])
        loss_func = solvers.LossFunction(
            dataset=ds, criterion=criterion, net=net_2
        )
        solver = solvers.GradientDescent(loss=loss_func, params=params, lr=0.1)
        loss_2 = torch.tensor([solver.step().loss], device=DEVICE)
        param_dict = _params_tensor_to_dict(params, loss_func._net)

        self.assertTrue(torch.isclose(loss_1, loss_2).all())
        for n, p_1 in net_1.named_parameters():
            self.assertTrue(torch.isclose(p_1, param_dict[n]).all())


if __name__ == '__main__':
    unittest.main()
