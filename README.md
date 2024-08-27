# Neural subdomain solver for magnetostatic field computations

This repository contains the code which implements a neural subdomain solver for magnetostatic field computations.
As a geometry mapping, we use AD-differentiable Non-Uniform Rational B-Splines (NURBS), which are implemented in the repository "./src/."
The differentiable NURBS can be used to induce a pullback mapping which allows to solve the magnetostatic field computation in a reference domain (unit hypercube).

## UNDER DEVELOPMENT

## Requirements & Install

The code in this repository is based on jax and flax to combine neural networks and numerical simulations.
The required packages to run the code in this repository is 

Mandatory:
- `JAX`: A Python library for accelerator-oriented array computation and program transformation. Useful for AD compatible transformations.
- `Flax`: A neural network library and ecosystem for JAX that is designed for flexibility.
- `Optax`: A a gradient processing and optimization library for JAX.
- `NumPy` Linear algebra backend.
- `Matplotlib`: Library for creating visualisations.
- `Pyvista`: A helper module for the Visualization Toolkit (VTK) that wraps the VTK library through NumPy and direct array access through a variety of methods and classes.

Optional for computing reference solutions:
- `FEniCS(x)`: An open-source computing platform for solving partial differential equations (PDEs) with the finite element method (FEM).

