import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers
import matplotlib as mpl
import matplotlib.pyplot as plt

import src 
import datetime
import jax.scipy.optimize
import jax.flatten_util
import scipy
import scipy.optimize
from mke_geo import *
from jax.config import config
config.update("jax_enable_x64", True)
rnd_key = jax.random.PRNGKey(1234)

def evaluate_models(model, params, ys, x):
    
    model.weights = params
    weights = params 
    u1 = model.solution1(weights, ys).reshape(x.shape)
    u2 = model.solution2(weights, ys).reshape(x.shape)
    u3 = model.solution3(weights, ys).reshape(x.shape)
    u4 = model.solution4(weights, ys).reshape(x.shape)
    u5 = model.solution5(weights, ys).reshape(x.shape)
    u6 = model.solution6(weights, ys).reshape(x.shape)
    u7 = model.solution7(weights, ys).reshape(x.shape)
    u8 = model.solution8(weights, ys).reshape(x.shape)

    vmin = min([u1.min(),u2.min(),u3.min(),u4.min(),u5.min(),u6.min(),u7.min(),u8.min()]) 
    vmax = max([u1.max(),u2.max(),u3.max(),u4.max(),u5.max(),u6.max(),u7.max(),u8.max()])
    return [u1, u2, u3, u4, u5, u6, u7, u8], vmin, vmax
                                                   


def evaluate_error(model, params):
    coordinates = np.loadtxt('/home/mvt/iga_pinns/fem_ref/coordinates.csv', delimiter = ',')
    ref_values = np.loadtxt('/home/mvt/iga_pinns/fem_ref/ref_values.csv', delimiter = ',')
    iron_pole, iron_yoke, iron_yoke_r_mid, iron_yoke_r_low, air_1, air_2, air_3, current  = create_geometry(rnd_key)
    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
    sol_model, vmin, vmax = evaluate_models(model, params, ys, x)

    vmin = np.amin(ref_values)
    vmax = np.amax(ref_values)
    vmin = 0
    vmax = 0.2 
    error = [] 
    print(vmin, vmax)
    plt.figure()
    norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap = 'viridis')
    m.set_array([])

    for i in range(8):
        step = 100**2
        local_coors = coordinates[i*step:(i+1)*step, :]
        local_vals = ref_values[i*step:(i+1)*step]
        local_x = local_coors[:,0]
        local_y = local_coors[:,1]
        xx = np.reshape(local_x, (100, 100))
        yy = np.reshape(local_y, (100, 100))
        uu = np.reshape(local_vals, (100, 100))
        error_local = np.abs(sol_model[i] - uu)
        error.append(np.sum(error_local))
        relative_error_domain = np.sum(error_local)/np.sum(np.abs(uu))
        print('The relative error in domain ', i + 1, ' is ', relative_error_domain*100, ' %')
        plt.contourf(xx,yy,error_local,norm = norm, levels = 100)
        plt.colorbar(m)
        plt.show()
        exit()
    plt.colorbar(m)
    plt.show()
    error = np.array(error)
    error_tot = np.sum(error)
    relative = error_tot/np.sum(np.abs(ref_values))
    print('The relative error amounts to ', relative*100, ' %')


    




