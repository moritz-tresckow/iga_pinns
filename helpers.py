import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers
import matplotlib.pyplot as plt
import src 
import datetime
import jax.scipy.optimize
import jax.flatten_util
import scipy
import scipy.optimize
from jax.config import config
config.update("jax_enable_x64", True)


def bnd_quadrature(dom, d, end, ys):
        def eval_dL(dom, d, input_vec):
            DGys = dom._eval_omega(input_vec)
            #diff = np.sqrt(DGys[:,0,d]**2 + DGys[:,1,d]**2)
            diff = [DGys[:,0,d], DGys[:,1,d]]
            return np.array(diff).T

        input_vec = end*jnp.ones((N,2))
        w_quad = jnp.ones((N,))*2/ys.shape[0]
        input_vec = input_vec.at[:,d].set(ys)
        diff = eval_dL(dom, d, input_vec)
        return input_vec, diff, w_quad
    
    ys = jnp.linspace(1,-1,N) 
    ys_right, diff_right, w_quad_right = bnd_quadrature(current, 1, 1, ys)
    ys_bottom, diff_bottom, w_quad_bottom = bnd_quadrature(current, 0, -1, ys)
    
    ys = jnp.linspace(-1,1,N) 
    ys_left, diff_left, w_quad_left = bnd_quadrature(current, 1, -1, ys)
    ys_top, diff_top, w_quad_top = bnd_quadrature(current, 0, 1, ys)

    ys_bnd = jnp.concatenate((ys_right, ys_bottom, ys_left, ys_top), axis = 0)

    diff = jnp.concatenate((diff_right, diff_bottom, diff_left, diff_top), axis = 0)
    w_quad = jnp.concatenate((w_quad_right, w_quad_bottom, w_quad_left, w_quad_top), axis = 0)
    
    points['ys_bnd8'] = ys_bnd
    points['omega_bnd8'] = diff
    points['ws_bnd8'] = w_quad#[:,jnp.newaxis]



def loss_constraint(self, ws, points):
    cc = src.operators.gradient(lambda x : model.solution8(ws,x))(points['ys_bnd8'])#[...,0,:]
    cc = jnp.concatenate((cc[:,:,1], -1*cc[:,:,0]), axis = 1)
    val = jnp.sum(jnp.sum(cc*points['omega_bnd8'], axis = 1) * points['ws_bnd8'])
    lpde_constr = jnp.abs(val - 1.12) 
    return lpde_constr



# lpde9 = jnp.abs(jnp.dot(self.J0*self.solution8(ws,points['ys8']).flatten()*points['omega8'] , points['ws8']) - 4.96104063)
