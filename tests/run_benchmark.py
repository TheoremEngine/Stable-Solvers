import argparse
import time
from typing import Callable, Iterable

from tabulate import tabulate
import torch

import stable_solvers as solvers


BATCH_SIZE = 1024
DATASET_SIZE = 4096
INPUT_DIM = 128
NETWORK_DEPTH = 6
NETWORK_WIDTH = 512
NUM_CLASSES = 4

# Tested on a LambdaLabs A100-SXM4 node
#
# ------  ---------
# g_eig   0.51178 sec
# grad    0.03340 sec
# h_eig   2.88250 sec
# hess    0.47660 sec
# solver  0.28432 sec
# ------  ---------


def time_it(func: Callable, args: Iterable, num_runs: int):
    total_time = 0

    for _ in range(num_runs):
        start_time = time.perf_counter()
        func(*args)
        torch.cuda.synchronize()
        total_time += (time.perf_counter() - start_time)

    return total_time / num_runs


def make_loss_func():
    torch.manual_seed(0)
    xs = torch.randn((DATASET_SIZE, INPUT_DIM), device='cuda')
    ys = torch.tensor(
        [i % NUM_CLASSES for i in range(DATASET_SIZE)], device='cuda'
    )
    ds = torch.utils.data.TensorDataset(xs, ys)
    criterion = torch.nn.CrossEntropyLoss()

    net = [torch.nn.Linear(INPUT_DIM, NETWORK_WIDTH)]
    for _ in range(NETWORK_DEPTH - 2):
        net += [torch.nn.ELU(), torch.nn.Linear(NETWORK_WIDTH, NETWORK_WIDTH)]
    net += [torch.nn.ELU(), torch.nn.Linear(NETWORK_WIDTH, NUM_CLASSES)]
    net = torch.nn.Sequential(*net).cuda()

    loss_func = solvers.LossFunction(
        ds, criterion, net=net, batch_size=BATCH_SIZE
    )
    params = loss_func.initialize_parameters(device='cuda')

    return loss_func, params


def make_solver():
    loss_func, params = make_loss_func()
    solver = solver.ExponentialEulerSolver(params, loss_func, 0.01, NUM_CLASSES)
    return (solver,)


def run_g_eig(loss_func, params):
    solver.g_eigenvector(loss_func, params, max_iters=1000)


def run_grad(loss_func, params):
    loss_func.gradient(params)


def run_h_eig(loss_func, params):
    solver.h_eigenvector(loss_func, params, max_iters=1000)


def run_hess(loss_func, params):
    solver.loss_hessian_eigenvector(loss_func, params, max_iters=1000)


def run_solver(solver):
    solver.step()


CASES = {
    'g_eig': (run_g_eig, make_loss_func),
    'grad': (run_grad, make_loss_func),
    'h_eig': (run_h_eig, make_loss_func),
    'hess': (run_hess, make_loss_func),
    'solver': (run_solver, make_solver),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('opts', nargs='*', choices=list(CASES.keys()) + [[]])
    args = parser.parse_args()

    if not args.opts:
        args.opts = list(CASES.keys())

    times = []
    for opt in args.opts:
        func, init = CASES[opt]
        times.append([opt, time_it(func, init(), 64)])

    print(tabulate(times))
