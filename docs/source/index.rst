.. Stable Solvers documentation master file, created by
   sphinx-quickstart on Mon Jul 29 21:01:55 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Stable Solvers
==============

Stable Solvers provides solvers for training neural networks without entering the edge of stability, for use in investigating neural network loss landscapes. As discovered in `Cohen et al. 2021 <https://arxiv.org/abs/2103.00065>`_, during neural network training, the curvature of the Hessian matrix rises until the training is overshooting. Stable Solvers provides adaptive solvers that avoid this overshooting and follow the true gradient flow.

The library currently provides two solvers: adaptive gradient descent and the exponential Euler solver. Adaptive gradient descent calculates the curvature of the Hessian matrix at every iteration and adjusts the learning rate to prevent overshooting. The exponential Euler solver exploits our knowledge of the quadratic terms of the loss function to take larger steps. Each step of the adaptive gradient method is cheaper, but it has to take more of them. Which is optimal depends on the problem to be solved: the exponential Euler method is generally best if the number of network outputs is small or the dataset size is very large, while the adaptive gradient method is best if neither of those conditions apply.

.. automodule:: stable_solvers
   :noindex:

.. currentmodule:: stable_solvers

.. autoclass:: LossFunction

   .. automethod:: __init__
   .. automethod:: forward
   .. automethod:: gradient
   .. automethod:: initialize_parameters

.. autoclass:: GradientDescent
   :members:

   .. automethod:: __init__

.. autoclass:: AdaptiveGradientDescent
   :members:

   .. automethod:: __init__

.. autoclass:: ExponentialEulerSolver
   :members:

   .. automethod:: __init__

