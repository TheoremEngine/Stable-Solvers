import unittest

import torch

import exponential_euler_solver as euler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class AutogradTest(unittest.TestCase):
    def test_mhp(self):
        def f(x):
            return (2 * x.pow(2)).sum()
        inputs = torch.rand(2, device=DEVICE)
        m = torch.randn(2, 3, device=DEVICE)

        mhp_1 = []
        for d in range(3):
            _, _mhp = torch.autograd.functional.vhp(f, inputs, m[:, d])
            mhp_1.append(_mhp)
        mhp_1 = [torch.stack(x, dim=-1) for x in zip(*mhp_1)]

        mhp_2 = euler.autograd.mhp(f, inputs, m)

        self.assertEqual(len(mhp_1), len(mhp_2))
        for _mhp_1, _mhp_2 in zip(mhp_1, mhp_2):
            self.assertEqual(_mhp_1.shape, _mhp_2.shape)
            self.assertTrue(torch.isclose(_mhp_1, _mhp_2).all())


if __name__ == '__main__':
    unittest.main()
