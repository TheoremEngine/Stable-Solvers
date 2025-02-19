## Stable Solvers <img align="right" src="docs/source/_static/logo.png" width=128 height=128>

This library provides solvers for training neural networks without entering the edge of stability. The edge of stability phenomenon was discovered in [Cohen et al. 2021](https://arxiv.org/abs/2103.00065). This repo is based on the approach used in [Lowell and Kastner 2024](https://arxiv.org/abs/2406.00127), but it is a complete reimplementation incorporating new techniques that improve runtime efficiency and support more recent versions of PyTorch. The purpose of this library is to support scientific research investigating the true gradient flow of on the loss landscape. These solvers are too computationally expensive to be used in practice. You can find documentation [here](https://www.theorem-engine.org/stable_solvers/). You can install this library by downloading this repo, or by running:

```
pip install stable-solvers
```

Training neural networks without entering the edge of stability requires departing somewhat from the conventional syntax in PyTorch, because the solver needs to have direct access to the loss function so it can calculate eigenvectors and eigenvalues. The recommended syntax in this repo is:

```
import stable_solvers as solvers

net = ...
dataset = ...
criterion = ...

loss_func = solvers.LossFunction(
    dataset=dataset,
    criterion=criterion,
    net=net,
    num_workers=1,
    batch_size=32
)
params = loss_func.initialize_parameters()
solver = solvers.ExponentialEulerSolver(
    params=params,
    loss=loss_func,
    max_step_size=0.01,
    stiff_dim=...  # Should be equal to the dimension of the network outputs
)
loss = float('inf')

while loss > 0.1:
    loss = solver.step().loss
```

A full example can be found in the [example notebook](Exponential-Euler-Solver-Example.ipynb).
