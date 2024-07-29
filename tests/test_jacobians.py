import unittest

import torch

import stable_solvers as solvers

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LayerwiseTest(unittest.TestCase):
    def test_layerwise_funcs(self):
        xs = torch.randn((5, 3), device=DEVICE)
        net = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.ReLU(False),
            torch.nn.Linear(8, 2)
        )
        net = net.to(DEVICE)

        # The true 0th layerwise Jacobian is just the weight of the linear
        # layer, by definition. The true 1st layerwise Jacobian is given by:
        #
        # \hat{x}^2 = A^2 \sigma( \hat{x}^1 ) + b^2
        # \frac{\partial \hat{x}^2}{\partial \hat{x}^1}_{i,j} =
        #     A^2_{i,j} \sigma'( \hat{x}^1_j )

        jac_1_0 = net[0].weight.T.unsqueeze(0)
        sigma_prime = (net[0](xs) > 0).float().unsqueeze(2)
        jac_1_1 = net[2].weight.T.unsqueeze(0) * sigma_prime

        params = torch.cat([p.flatten() for p in net.parameters()])
        jac_func = solvers.make_layerwise_jacobian_function(net, params)
        jac_2_0, jac_2_1 = jac_func(xs)

        self.assertTrue(torch.allclose(jac_1_0, jac_2_0))
        self.assertTrue(torch.allclose(jac_1_1, jac_2_1))


if __name__ == '__main__':
    unittest.main()
