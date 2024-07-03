import unittest

import torch

import exponential_euler_solver as euler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LossFunctionTest(unittest.TestCase):
    def test_criterion(self):
        def criterion(x, y):
            return (x - y).abs().sum()

        xs = torch.randn((4, 3), device=DEVICE)
        ys = torch.tensor([0, 0, 1, 1], device=DEVICE, dtype=torch.int64)
        ds = torch.utils.data.TensorDataset(xs, ys)
        loss_func = euler.LossFunction(
            dataset=ds, criterion=criterion, net=torch.nn.Linear(4, 4),
        )

        with self.assertRaises(RuntimeError):
            loss_func.criterion(ys, ys)

    def test_forward(self):
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

        out = net(xs)
        loss_1 = criterion(out, ys).mean()

        params = torch.cat([p.flatten() for p in net.parameters()])
        loss_func = euler.LossFunction(
            dataset=ds, criterion=criterion, net=net
        )
        loss_2 = loss_func(params)

        self.assertTrue(torch.isclose(loss_1, loss_2).all())

    def test_gradient(self):
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

        out = net(xs)
        loss_1 = criterion(out, ys).mean()
        loss_1.backward()
        grads_1 = [p.grad for p in net.parameters()]

        params = torch.cat([p.flatten() for p in net.parameters()])
        loss_func = euler.LossFunction(
            dataset=ds, criterion=criterion, net=net
        )
        loss_2, grads_2 = loss_func.gradient(params)
        grads_2 = loss_func._to_dict(grads_2)

        self.assertTrue(torch.isclose(loss_1, loss_2).all())
        for g, (n, _) in zip(grads_1, net.named_parameters()):
            self.assertTrue(torch.isclose(g, grads_2[n]).all())

    def test_initialize_parameters(self):
        xs = torch.randn((4, 3), device=DEVICE)
        ys = torch.tensor([0, 0, 1, 1], device=DEVICE, dtype=torch.int64)
        ds = torch.utils.data.TensorDataset(xs, ys)
        criterion = torch.nn.CrossEntropyLoss()
        net = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.ReLU(True),
            torch.nn.Linear(8, 2)
        )
        loss_func = euler.LossFunction(
            dataset=ds, criterion=criterion, net=net
        )
        params = loss_func.initialize_parameters(device='cuda')
        param_dict = loss_func._to_dict(params)
        self.assertEqual(
            param_dict.keys(), {'0.weight', '0.bias', '2.weight', '2.bias'}
        )
        self.assertTrue((param_dict['0.bias'] == 0).all())
        self.assertTrue((param_dict['2.bias'] == 0).all())


if __name__ == '__main__':
    unittest.main()
