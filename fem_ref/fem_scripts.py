#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: momo_the_destroyer
"""
import sys
from fenics import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from quad_param import load_mesh
np.set_printoptions(threshold=sys.maxsize)

# mu0=4*np.pi*1e-7
# mur = 1000


mu0 = 1
curr = 1000
def assemble_fem(mesh, domains, boundaries, mur, curr, idx):
    vertex  = mesh.coordinates()
    CG = FunctionSpace(mesh, 'CG', 1) # Continuous Galerkin
    # Define boundary condition
    bc = DirichletBC(CG, Constant(0.0), boundaries, 16)
    # Define subdomain markers and integration measure
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    DG = FunctionSpace(mesh,"DG",0)
    J = Function(DG)
    cells_idx = domains.where_equal(idx)
    J.vector()[:] = 0
    J.vector()[cells_idx] = curr

    class Nu(UserExpression): # UserExpression instead of Expression
        def __init__(self, markers, **kwargs):
            super().__init__(**kwargs) # This part is new!
            self.markers = markers
        def eval_cell(self, values, x, cell):
            if self.markers[cell.index] == 2:
               values[0] = 1/mu0 
            elif self.markers[cell.index] == 1:
                values[0] = 1/(mu0*mur)
            elif self.markers[cell.index] == 3:
                values[0] = 1/(mu0)
            else:
                print('no such domain')
    
    nu = Nu(domains, degree=1)
    v  = TrialFunction(CG)
    u  = TestFunction(CG)
    a  = nu*inner(grad(u), grad(v))*dx
    L  = J*u*dx(3)
    uh = Function(CG)
    solve(a == L, uh, bc)
    sol = uh.vector()[:]
    vmin = np.amin(sol)
    vmax = np.amax(sol)
    norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap = 'viridis')
    m.set_array([])
    print('The minimum is ', vmin, ' and the maximum is ', vmax)
    plot(uh)
    plt.colorbar(m)
    plt.savefig('./my-fig.png')
    return sol, uh



mesh, domains, boundaries = load_mesh('./quad_simple')
sol_curr, uh = assemble_fem(mesh, domains, boundaries, 2000, curr, 3)

coordinates = np.loadtxt('./coordinates_simple.csv', delimiter = ',')
ref_values = np.ones((coordinates.shape[0],1))
for i in range(coordinates.shape[0]):
    x = Point(coordinates[i,0], coordinates[i,1])
    ref_values[i] = uh(x)
np.savetxt('./ref_values_simple.csv', ref_values, delimiter = ',', comments = '')
exit()
ref_values = np.loadtxt('./ref_values.csv', delimiter = ',')
vmin = np.amin(ref_values)
vmax = np.amax(ref_values)

for i in range(8):
    step = 100**2
    local_coors = coordinates[i*step:(i+1)*step, :]
    local_vals = ref_values[i*step:(i+1)*step]
    local_x = local_coors[:,0]
    local_y = local_coors[:,1]
    xx = np.reshape(local_x, (100, 100))
    yy = np.reshape(local_y, (100, 100))
    uu = np.reshape(local_vals, (100, 100))
    plt.contourf(xx,yy,uu,vmin = vmin, vmax = vmax, levels = 100)
plt.show()
