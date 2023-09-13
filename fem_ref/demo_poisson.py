from dolfin import *
from quad_param import load_mesh
import numpy as np


mesh, domains, boundaries = load_mesh('./cyl')

V = FunctionSpace(mesh, "Lagrange", 1)

bc = DirichletBC(V, Constant(0), boundaries, 2)
bcs = [bc]

u = TrialFunction(V)
v = TestFunction(V)
f = 1 
a = inner(grad(u), grad(v))*dx
L = f*v*dx
# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# Plot solution
vmin = np.amin(u.vector()[:])
vmax = np.amax(u.vector()[:])
print(vmin, vmax)
import matplotlib.pyplot as plt
plot(u)
plt.show()
