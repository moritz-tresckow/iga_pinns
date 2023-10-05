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


def assemble_fem(mesh, domains, boundaries, mu, curr, idx_d):
    def curl2D(v):
        return as_vector((v.dx(1),-v.dx(0)))
    def curl2D_H(h):
        return h[1].dx(0)-h[0].dx(1)

    idx_b = idx_d[0]
    idx_c = idx_d[1]
    mu0 = mu[0]
    mur = mu[1]
    vertex  = mesh.coordinates()
    CG = FunctionSpace(mesh, 'CG', 1) # Continuous Galerkin
    # Define boundary condition
    bc = DirichletBC(CG, Constant(0.0), boundaries, idx_b)
    # Define subdomain markers and integration measure
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    DG = FunctionSpace(mesh,"DG",0)
    J = Function(DG)
    cells_idx = domains.where_equal(idx_c)
    J.vector()[:] = 0
    J.vector()[cells_idx] = curr

    class Nu(UserExpression): # UserExpression instead of Expression
        def __init__(self, markers, **kwargs):
            super().__init__(**kwargs) # This part is new!
            self.markers = markers
        def eval_cell(self, values, x, cell):
            if self.markers[cell.index] == 1:
               values[0] = 1/mu0 
            elif self.markers[cell.index] == 2:
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
    # ds = Measure('ds', domain = mesh, subdomain_data = boundaries, subdomain_id=6)
    uh = Function(CG)
    solve(a == L, uh, bc)

    plt.savefig('./my-fig.png')
    print(assemble(uh*J*dx(3)))
    
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

def cal_ref_with_coors(uh, path):
    coordinates = np.loadtxt('./coordinates.csv', delimiter = ',')
    ref_values = np.ones((coordinates.shape[0],1))
    for i in range(coordinates.shape[0]):
        x = Point(coordinates[i,0], coordinates[i,1])
        ref_values[i] = uh(x)
    np.savetxt(path + 'ref_values_950.csv', ref_values, delimiter = ',', comments = '')



mesh, domains, boundaries = load_mesh('./quad')
mu0 = 1
mur = 1 
mu = [mu0, mur]
curr = 950
idx = [5,3]
sol_curr, uh = assemble_fem(mesh, domains, boundaries, mu, curr, idx)
cal_ref_with_coors(uh, '/Users/moritzvontresckow/Desktop/iga_pinns/parameters/quad/no_mat/')
idx_simple = [16,3]