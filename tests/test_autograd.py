import unittest

import torch

import exponential_euler_solver as euler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class AutogradTest(unittest.TestCase):
    def test_mhp(self):
        def pow_adder_reducer(x, y):
            return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        inputs = (torch.rand(2, device=DEVICE), torch.rand(2, device=DEVICE))
        m = (
            torch.randn(2, 3, device=DEVICE),
            torch.randn(2, 3, device=DEVICE)
        )

        mhp_1 = []
        for d in range(3):
            v = tuple(x[:, d] for x in m)
            _, _mhp = torch.autograd.functional.vhp(
                pow_adder_reducer, inputs, v
            )
            mhp_1.append(_mhp)
        mhp_1 = [torch.stack(x, dim=-1) for x in zip(*mhp_1)]

        mhp_2 = euler.autograd.mhp(pow_adder_reducer, inputs, m)

        self.assertEqual(len(mhp_1), len(mhp_2))
        for _mhp_1, _mhp_2 in zip(mhp_1, mhp_2):
            self.assertEqual(_mhp_1.shape, _mhp_2.shape)
            self.assertTrue(torch.isclose(_mhp_1, _mhp_2).all())

        mhp_2 = euler.autograd.mhp(
            pow_adder_reducer, inputs, (m[0][:, 0:1], m[1][:, 0:1])
        )
        for _mhp_1, _mhp_2 in zip(mhp_1, mhp_2):
            self.assertTrue(torch.isclose(_mhp_1[..., 0:1], _mhp_2).all())


if __name__ == '__main__':
    unittest.main()
