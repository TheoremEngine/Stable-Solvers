This repo provides an [exponential Euler solver](https://en.wikipedia.org/wiki/Exponential_integrator) and related utilities for training neural networks without entering the edge of stability. The edge of stability phenomenon was discovered in [Cohen et al. 2021](https://arxiv.org/abs/2103.00065). This repo is based on the approach used in [Lowell and Kastner 2024](https://arxiv.org/abs/2406.00127), but it is a complete reimplementation incorporating new techniques that improve runtime efficiency and support more recent versions of PyTorch. **IT IS STILL UNDER HEAVY CONSTRUCTION**.

Training neural networks with an exponential Euler solver requires departing somewhat from the conventional syntax in PyTorch, because the solver needs to have direct access to the loss function so it can calculate eigenvectors and eigenvalues. The recommended syntax in this repo is:

```
import exponential_euler_solver as euler

net = ...
dataset = ...
criterion = ...

loss_func = euler.LossFunction(
    dataset=dataset,
    criterion=criterion,
    net=net,
    num_workers=1,
    batch_size=32
)
params = loss_func.initialize_parameters()
solver = euler.ExponentialEulerSolver(
    params=params,
    loss=loss_func,
    max_step_size=0.01,
    stiff_dim=...  # Should be equal to the dimension of the network outputs
)
loss = float('inf')

while loss > 0.1:
    loss = solver.step().loss
```

**WARNING:** Training using an exponential Euler solver is *extremely* computationally intensive, due to the need to solve for eigenvectors of the loss Hessian at every iteration of training.
