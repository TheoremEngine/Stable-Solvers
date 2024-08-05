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
        grad = torch.cat(
            [p.grad.flatten().detach() for p in net_1.parameters()]
        )
        opt.step()

        params = torch.cat([p.flatten() for p in net_2.parameters()])
        loss_func = solvers.LossFunction(
            dataset=ds, criterion=criterion, net=net_2
        )
        solver = solvers.GradientDescent(
            loss=loss_func, params=params, lr=0.1, report_gradient=True,
        )
        report = solver.step()
        loss_2 = torch.tensor([report.loss], device=DEVICE)
        param_dict = _params_tensor_to_dict(params, loss_func._net)

        self.assertTrue(torch.isclose(loss_1, loss_2).all())
        for n, p_1 in net_1.named_parameters():
            self.assertTrue(torch.isclose(p_1, param_dict[n]).all())
        self.assertTrue(torch.isclose(report.gradient, grad).all())

        solver.report_gradient = False
        self.assertTrue(solver.step().gradient is None)

    def test_adaptive_gradient_descent(self):
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

        params = torch.cat([p.flatten() for p in net_2.parameters()])
        loss_func = solvers.LossFunction(
            dataset=ds, criterion=criterion, net=net_2
        )
        solver = solvers.AdaptiveGradientDescent(
            loss=loss_func, params=params, lr=0.1, report_gradient=True,
            report_eigvec=True,
        )
        report = solver.step()

        lr = min(0.1, 1. / report.sharpness)
        opt = torch.optim.SGD(net_1.parameters(), lr=lr, momentum=0.)
        out = net_1(xs)
        loss_1 = criterion(out, ys).mean()
        loss_1.backward()
        grad = torch.cat(
            [p.grad.flatten().detach() for p in net_1.parameters()]
        )
        opt.step()

        loss_2 = torch.tensor([report.loss], device=DEVICE)
        param_dict = _params_tensor_to_dict(params, loss_func._net)

        self.assertTrue(torch.isclose(loss_1, loss_2).all())
        for n, p_1 in net_1.named_parameters():
            self.assertTrue(torch.isclose(p_1, param_dict[n]).all())
        self.assertTrue(torch.isclose(report.gradient, grad).all())

        solver.report_gradient = solver.report_eigvec = False
        report = solver.step()
        self.assertTrue(report.gradient is None)
        self.assertTrue(report.eigvec is None)

    def test_check_loss(self):
        criterion = torch.nn.CrossEntropyLoss()
        xs = torch.randn((4, 3), device=DEVICE)
        ys = torch.tensor([0, 0, 1, 1], device=DEVICE, dtype=torch.int64)
        ds = torch.utils.data.TensorDataset(xs, ys)

        net = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.ReLU(True),
            torch.nn.Linear(8, 2)
        )
        net = net.to(DEVICE)

        loss_func = solvers.LossFunction(
            dataset=ds, criterion=criterion, net=net
        )
        params = loss_func.initialize_parameters(device=DEVICE)
        solver = solvers.AdaptiveGradientDescent(
            loss=loss_func, params=params, lr=0.01, error_if_unstable=True,
        )
        solver.last_loss = 0.

        with self.assertRaises(RuntimeError):
            solver.step()

    def test_exponential_euler_solver(self):
        criterion = torch.nn.CrossEntropyLoss()
        xs = torch.randn((4, 3), device=DEVICE)
        ys = torch.tensor([0, 0, 1, 1], device=DEVICE, dtype=torch.int64)
        ds = torch.utils.data.TensorDataset(xs, ys)

        net = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.ReLU(True),
            torch.nn.Linear(8, 2)
        )
        net = net.to(DEVICE)

        loss_func = solvers.LossFunction(
            dataset=ds, criterion=criterion, net=net
        )
        params = loss_func.initialize_parameters(device=DEVICE)
        solver = solvers.ExponentialEulerSolver(
            loss=loss_func, params=params, max_step_size=0.1, stiff_dim=1,
            report_gradient=True, report_eigvecs=True,
        )
        report = solver.step()
        self.assertTrue(report.gradient is not None)
        self.assertEqual(report.gradient.shape, (50,))
        self.assertTrue(report.eigvecs is not None)
        self.assertEqual(report.eigvecs.shape, (50, 2))

        solver.report_gradient = solver.report_eigvecs = False
        report = solver.step()
        self.assertTrue(report.gradient is None)
        self.assertTrue(report.eigvecs is None)


if __name__ == '__main__':
    unittest.main()
