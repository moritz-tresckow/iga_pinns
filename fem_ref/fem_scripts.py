#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: momo_the_destroyer
"""
import sys
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from quad_param import load_mesh
np.set_printoptions(threshold=sys.maxsize)


mu0=4*np.pi*1e-7
mur = 1000
def assemble_fem(mesh, domains, boundaries, mur, idx):
    vertex  = mesh.coordinates()
    CG = FunctionSpace(mesh, 'CG', 1) # Continuous Galerkin
    # Define boundary condition
    bc = DirichletBC(CG, Constant(0.0), boundaries, 5)
    # Define subdomain markers and integration measure
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    DG = FunctionSpace(mesh,"DG",0)
    J = Function(DG)
    cells_idx = domains.where_equal(idx)
    J.vector()[:] = 0
    J.vector()[cells_idx] = 1e3

    class Permeability(UserExpression): # UserExpression instead of Expression
        def __init__(self, markers, **kwargs):
            super().__init__(**kwargs) # This part is new!
            self.markers = markers
        def eval_cell(self, values, x, cell):
            if self.markers[cell.index] == 1:
               values[0] = mu0 
            elif self.markers[cell.index] == 2:
                values[0] = mu0*mur
            elif self.markers[cell.index] == 3:
                values[0] = mu0
            else:
                print('no such domain')
    
    # mu = Permeability(domains, degree=1)
    
    v  = TrialFunction(CG)
    u  = TestFunction(CG)
    a  = inner(grad(u), grad(v))*dx
    L  = J*u*dx(3)
    uh = Function(CG)
    solve(a == L, uh, bc)
    print(uh.vector()[:])
    plot(uh)
    plt.savefig('./myfig.png')
    exit()
    A,b = assemble_system(a, L, bc)
    return A, b, vertex



mesh, domains, boundaries = load_mesh('/Users/moritzvontresckow/Desktop/iga_pinns/fem_ref/quad')
print('Hello!')
assemble_fem(mesh, domains, boundaries, 1, 3)