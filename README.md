# Neural subdomain solver for magnetostatic field computations

This repository contains the code which implements a neural subdomain solver for magnetostatic field computations.
As a geometry mapping, we use AD-differentiable Non-Uniform Rational B-Splines (NURBS), which are implemented in the repository "./src/."
The differentiable NURBS can be used to induce a pullback mapping which allows to solve the magnetostatic field computation in a reference domain (unit hypercube).


## Requirements & Install

The code in this repository is based on jax and flax to combine neural networks and numerical simulations.
The required packages to run the code in this repository is 

- `jax`
- `flax`
- `optax`
