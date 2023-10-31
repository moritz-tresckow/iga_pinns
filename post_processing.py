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
from fenicsx_scripts import calc_eq
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

def evaluate_models5(model, params, ys, x):
    model.weights = params
    weights = params 
    u1 = model.solution1(weights, ys).reshape(x.shape)
    u2 = model.solution2(weights, ys).reshape(x.shape)
    u3 = model.solution3(weights, ys).reshape(x.shape)
    u4 = model.solution4(weights, ys).reshape(x.shape)
    u5 = model.solution5(weights, ys).reshape(x.shape)

    vmin = min([u1.min(),u2.min(),u3.min(),u4.min(),u5.min()]) 
    vmax = max([u1.max(),u2.max(),u3.max(),u4.max(),u5.max()])
    return [u1, u2, u3, u4, u5], vmin, vmax

def evaluate_models7(model, params, ys, x):
    model.weights = params
    weights = params 
    u1 = model.solution1(weights, ys).reshape(x.shape)
    u2 = model.solution2(weights, ys).reshape(x.shape)
    u3 = model.solution3(weights, ys).reshape(x.shape)
    u4 = model.solution4(weights, ys).reshape(x.shape)
    u5 = model.solution5(weights, ys).reshape(x.shape)
    u6 = model.solution6(weights, ys).reshape(x.shape)
    u7 = model.solution7(weights, ys).reshape(x.shape)

    vmin = min([u1.min(),u2.min(),u3.min(),u4.min(),u5.min(),u6.min(),u7.min()]) 
    vmax = max([u1.max(),u2.max(),u3.max(),u4.max(),u5.max(),u6.max(),u7.max()])
    return [u1, u2, u3, u4, u5, u6, u7], vmin, vmax

def evaluate_air(model, params, ys, x):
    model.weights = params
    weights = params 
    #u1 = model.solution1(weights, ys).reshape(x.shape)
    #u2 = model.solution2(weights, ys).reshape(x.shape)
    #u3 = model.solution3(weights, ys).reshape(x.shape)
    #u4 = model.solution4(weights, ys).reshape(x.shape)
    u1=np.array(0)
    u2=np.array(0)
    u3=np.array(0)
    u4=np.array(0)
    u5 = model.solution5(weights, ys).reshape(x.shape)
    u6 = model.solution6(weights, ys).reshape(x.shape)
    u7 = model.solution7(weights, ys).reshape(x.shape)
    u8 = model.solution8(weights, ys).reshape(x.shape)

    vmin = min([u1.min(),u2.min(),u3.min(),u4.min(),u5.min(),u6.min(),u7.min(),u8.min()]) 
    vmax = max([u1.max(),u2.max(),u3.max(),u4.max(),u5.max(),u6.max(),u7.max(),u8.max()])
    return [u1, u2, u3, u4, u5, u6, u7, u8], vmin, vmax
        


def evaluate_single_model(model, params, ys, x):
    model.weights = params
    weights = params 
    u1 = model.solution1(weights, ys).reshape(x.shape)
    u2 = model.solution2(weights, ys).reshape(x.shape)
    vmin = min([u1.min(),u2.min()]) 
    vmax = max([u1.max(),u2.max()])
    return [u1, u2], vmin, vmax

def evaluate_triple_model(model, params, ys, x):
    model.weights = params
    weights = params 
    u1 = model.solution1(weights, ys).reshape(x.shape)
    u2 = model.solution2(weights, ys).reshape(x.shape)
    u3 = model.solution3(weights, ys).reshape(x.shape)
    vmin = min([u1.min(),u2.min(), u3.min()]) 
    vmax = max([u1.max(),u2.max(), u3.max()])
    return [u1, u2, u3], vmin, vmax



def evaluate_quad_double_model(model, params, ys, x):
    model.weights = params
    weights = params 
    u1 = model.solution1(weights, ys).reshape(x.shape)
    u4 = model.solution4(weights, ys).reshape(x.shape)
    vmin = min([u1.min(),u4.min()]) 
    vmax = max([u1.max(),u4.max()])
    return [u1, u4], vmin, vmax



def evaluate_quad_nonlin(model, params, ys, x):
    model.weights = params
    weights = params 
    u1 = model.solution1(weights, ys).reshape(x.shape)
    u2 = model.solution2(weights, ys).reshape(x.shape)
    u3 = model.solution3(weights, ys).reshape(x.shape)
    u4 = model.solution4(weights, ys).reshape(x.shape)

    vmin = min([u1.min(),u2.min(),u3.min(),u4.min()]) 
    vmax = max([u1.max(),u2.max(),u3.max(),u4.max()])
    return [u1, u2, u3, u4], vmin, vmax


def evaluate_quad_new(model, params, ys, x):
    model.weights = params
    weights = params 
    u1 = model.solution1(weights, ys).reshape(x.shape)
    u2 = model.solution2(weights, ys).reshape(x.shape)
    u3 = model.solution3(weights, ys).reshape(x.shape)

    vmin = min([u1.min(),u2.min(),u3.min()]) 
    vmax = max([u1.max(),u2.max(),u3.max()])
    return [u1, u2, u3], vmin, vmax

def cal_L2_error(ref_val, cal_val, msh):
    difference = (cal_val - ref_val)**2
    error = cal_functional(difference, msh)
    relative = cal_functional(ref_val, msh)
    return 0


def cal_coordinates(geoms):
    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
    outputs = []
    for i in geoms:
        out = i.__call__(ys)
        outputs.append(out)
    outputs = np.array(outputs)
    outputs = np.reshape(outputs, (outputs.shape[0]*outputs.shape[1],2))
    return outputs



def evaluate_error(model, params, evaluation_func, model_idxs, geoms, meshfile):
    coordinates = cal_coordinates(geoms)
    ref_values = calc_eq(meshfile, [model.mu0, model.mur], model.J0, coordinates)

    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
    sol_model, vmin, vmax = evaluation_func(model, params, ys, x)
    print('The min and max of the NN models is: ', vmin, vmax)

    #vmin = np.amin(ref_values)
    #vmax = np.amax(ref_values)
    #print('The min and max of the reference is: ', vmin, vmax)

    #vmin = 0
    #vmax = 0.1 
    error = [] 
    plt.figure()
    norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap = 'viridis')
    m.set_array([])

    for i in model_idxs:
        step = 100**2
        local_coors = coordinates[i*step:(i+1)*step, :]
        local_vals = ref_values[i*step:(i+1)*step]
        local_x = local_coors[:,0]
        local_y = local_coors[:,1]
        xx = np.reshape(local_x, (100, 100))
        yy = np.reshape(local_y, (100, 100))
        uu = np.reshape(local_vals, (100, 100))
        error_local = np.abs(sol_model[i] - uu)
        print(i+1, np.sum(error_local))
        print(i+1, np.sum(uu))
        error.append(np.sum(error_local))
        relative_error_domain = np.sum(error_local)/np.sum(np.abs(uu))
        print('The relative error in domain ', i + 1, ' is ', relative_error_domain*100, ' %')
        plt.contourf(xx, yy, sol_model[i], norm = norm, levels = 100)
        #plt.contourf(xx, yy, error_local, norm = norm, levels = 100)
        #plt.contourf(xx, yy, uu, norm = norm, levels = 100)
    plt.colorbar(m)
    # plt.show()
    plt.savefig('./complete_fig.png')
    error_tot = np.sum(error)
    relative = error_tot/np.sum(np.abs(ref_values))
    print('The relative error amounts to ', relative*100, ' %')


    


      



