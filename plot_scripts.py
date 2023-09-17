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
rnd_key = jax.random.PRNGKey(1234)

def plot_solution_quad_nonlin(model, params, geoms):
    geom1 = geoms[0]
    geom2 = geoms[1]
    geom3 = geoms[2]
    geom4 = geoms[3]
    model.weights = params
    weights = params

    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
    xy1 = geom1(ys)
    xy2 = geom2(ys)
    xy3 = geom3(ys)
    xy4 = geom4(ys)

    u1 = model.solution1(weights, ys).reshape(x.shape)
    u2 = model.solution2(weights, ys).reshape(x.shape)
    u3 = model.solution3(weights, ys).reshape(x.shape)
    u4 = model.solution4(weights, ys).reshape(x.shape)

    u1 = np.abs(u1)
    u2 = np.abs(u2)
    u3 = np.abs(u3)
    u4 = np.abs(u4)
    vmin = min([u1.min(),u2.min(),u3.min(),u4.min()])
    vmax = max([u1.max(),u2.max(),u3.max(),u4.max()])

    plt.figure(figsize = (20,12))
    ax = plt.gca()
    plt.contourf(xy1[:,0].reshape(x.shape), xy1[:,1].reshape(x.shape), u1, levels = 100, vmin = vmin, vmax = vmax)
    plt.contourf(xy2[:,0].reshape(x.shape), xy2[:,1].reshape(x.shape), u2, levels = 100, vmin = vmin, vmax = vmax)
    plt.contourf(xy3[:,0].reshape(x.shape), xy3[:,1].reshape(x.shape), u3, levels = 100, vmin = vmin, vmax = vmax)
    plt.contourf(xy4[:,0].reshape(x.shape), xy4[:,1].reshape(x.shape), u4, levels = 100, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.xlabel(r'$x_1$ [m]')
    plt.ylabel(r'$x_2$ [m]')
    plt.show()
    # plt.savefig('./myfig.png')


"""
knots_bottom = np.array([[a3[0],0], [d4x+offset,0]])
knots_bottom = np.array([knots_bottom[0], 0.5*(knots_bottom[0] + knots_bottom[1]), knots_bottom[1]])
knots_middle = np.array([a3, [0.0912132, 0.0243934]] ) 
knots_middle = np.array([knots_middle[0], 0.5*(knots_middle[0] + knots_middle[1]), knots_middle[1]])
knots_air1 = np.array([[d4x+offset,d4y+offset], k1])
knots_air1 = np.array([knots_air1[0], 0.5*(knots_air1[0] + knots_air1[1]), knots_air1[1]])
"""

def plot_bndr_quad_nonlin(model, weights):
    def sample_bnd():
        ys = np.linspace(-1,1,1000)
        ys = ys[:,np.newaxis]
        one_vec = np.ones_like(ys)

        ys_top = np.concatenate((ys, one_vec), axis = 1)
        ys_bottom = np.concatenate((ys, -1 * one_vec), axis = 1)

        ys_left = np.concatenate((-1* one_vec, ys), axis = 1) 
        ys_right = np.concatenate((one_vec, ys), axis = 1) 
        return ys, [ys_top, ys_right, ys_bottom, ys_left] 

    ys, samples = sample_bnd()
    vals1 = [model.solution1(weights, i) for i in samples]
    vals2 = [model.solution2(weights, i) for i in samples]
    vals3 = [model.solution3(weights, i) for i in samples]
    vals4 = [model.solution4(weights, i) for i in samples]
    
    
    plt.figure()
    plt.plot(ys, vals1[0], label = 'u12')
    plt.plot(ys, vals2[1], label = 'u21')
    plt.legend()
   
    plt.figure()
    plt.plot(ys, vals1[1], label = 'u13')
    plt.plot(ys, vals3[1], label = 'u51')
    plt.legend()
    
    plt.figure()
    plt.plot(ys, vals1[2], label = 'u14')
    plt.plot(ys, vals4[0], label = 'u51')
    plt.legend()

    plt.show()
    exit()