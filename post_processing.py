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
    u9 = model.solution9(weights, ys).reshape(x.shape)

    vmin = min([u1.min(),u2.min(),u3.min(),u4.min(),u5.min(),u6.min(),u7.min(),u8.min(),u9.min()]) 
    vmax = max([u1.max(),u2.max(),u3.max(),u4.max(),u5.max(),u6.max(),u7.max(),u8.max(), u9.max()])
    return [u1, u2, u3, u4, u5, u6, u7, u8, u9], vmin, vmax

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


def cal_l2_error(model, ws, geoms, meshfile):

    batch_size = 10000
    key = jax.random.PRNGKey(1235)
    points = model.get_points_MC(batch_size, key)

    coors1 = geoms[0].__call__(points['ys1'])
    coors2 = geoms[1].__call__(points['ys2'])
    coors3 = geoms[2].__call__(points['ys3'])
    coors4 = geoms[3].__call__(points['ys4'])
    coors5 = geoms[4].__call__(points['ys5'])
    coors6 = geoms[5].__call__(points['ys6'])
    coors7 = geoms[6].__call__(points['ys7'])
    coors8 = geoms[7].__call__(points['ys8'])
    coors9 = geoms[8].__call__(points['ys9'])
    
    ref_values_1 = calc_eq(meshfile, [model.mu0, model.mur], model.J0, coors1)
    model_values_1 = model.solution1(ws, points['ys1'])
    error_1 = ((ref_values_1 - model_values_1)**2).flatten()
    int_error_1 = jnp.dot(error_1*points['omega1']  ,points['ws1'])
    int_ref_1 =  jnp.dot(((ref_values_1)**2).flatten()*points['omega1']  ,points['ws1'])

    ref_values_2 = calc_eq(meshfile, [model.mu0, model.mur], model.J0, coors2)
    model_values_2 = model.solution2(ws, points['ys2'])
    error_2 = ((ref_values_2 - model_values_2)**2).flatten()
    int_error_2 = jnp.dot(error_2*points['omega2']  ,points['ws2'])
    int_ref_2 =  jnp.dot(((ref_values_2)**2).flatten()*points['omega2']  ,points['ws2'])

    ref_values_3 = calc_eq(meshfile, [model.mu0, model.mur], model.J0, coors3)
    model_values_3 = model.solution3(ws, points['ys3'])
    error_3 = ((ref_values_3 - model_values_3)**2).flatten()
    int_error_3 = jnp.dot(error_3*points['omega3']  ,points['ws3'])
    int_ref_3 =  jnp.dot(((ref_values_3)**2).flatten()*points['omega3']  ,points['ws3'])

    ref_values_4 = calc_eq(meshfile, [model.mu0, model.mur], model.J0, coors4)
    model_values_4 = model.solution4(ws, points['ys4'])
    error_4 = ((ref_values_4 - model_values_4)**2).flatten()
    int_error_4 = jnp.dot(error_4*points['omega4']  ,points['ws4'])
    int_ref_4 =  jnp.dot(((ref_values_4)**2).flatten()*points['omega4']  ,points['ws4'])

    ref_values_5 = calc_eq(meshfile, [model.mu0, model.mur], model.J0, coors5)
    model_values_5 = model.solution5(ws, points['ys5'])
    error_5 = ((ref_values_5 - model_values_5)**2).flatten()
    int_error_5 = jnp.dot(error_5*points['omega5']  ,points['ws5'])
    int_ref_5 =  jnp.dot(((ref_values_5)**2).flatten()*points['omega5']  ,points['ws5'])

    ref_values_6 = calc_eq(meshfile, [model.mu0, model.mur], model.J0, coors6)
    model_values_6 = model.solution6(ws, points['ys6'])
    error_6 = ((ref_values_6 - model_values_6)**2).flatten()
    int_error_6 = jnp.dot(error_6*points['omega6']  ,points['ws6'])
    int_ref_6 =  jnp.dot(((ref_values_6)**2).flatten()*points['omega6']  ,points['ws6'])

    ref_values_7 = calc_eq(meshfile, [model.mu0, model.mur], model.J0, coors7)
    model_values_7 = model.solution7(ws, points['ys7'])
    error_7 = ((ref_values_7 - model_values_7)**2).flatten()
    int_error_7 = jnp.dot(error_7*points['omega7']  ,points['ws7'])
    int_ref_7 =  jnp.dot(((ref_values_7)**2).flatten()*points['omega7']  ,points['ws7'])

    ref_values_8 = calc_eq(meshfile, [model.mu0, model.mur], model.J0, coors8)
    model_values_8 = model.solution8(ws, points['ys8'])
    error_8 = ((ref_values_8 - model_values_8)**2).flatten()
    int_error_8 = jnp.dot(error_8*points['omega8']  ,points['ws8'])
    int_ref_8 =  jnp.dot(((ref_values_8)**2).flatten()*points['omega8']  ,points['ws8'])

    ref_values_9 = calc_eq(meshfile, [model.mu0, model.mur], model.J0, coors9)
    model_values_9 = model.solution9(ws, points['ys9'])
    error_9 = ((ref_values_9 - model_values_9)**2).flatten()
    int_error_9 = jnp.dot(error_9*points['omega9']  ,points['ws9'])
    int_ref_9 =  jnp.dot(((ref_values_9)**2).flatten()*points['omega9']  ,points['ws9'])

    int_error = int_error_1 + int_error_2 + int_error_3 + int_error_4 + int_error_5 + int_error_6 + int_error_7 + int_error_8 + int_error_9 
    int_ref =  int_ref_1 + int_ref_2 + int_ref_3 + int_ref_4 + int_ref_5 + int_ref_6 + int_ref_7 + int_ref_8 + int_ref_9    


    print('Domain 1: ',np.sqrt(int_error_1/int_ref_1)*100, '%')
    print('Domain 2: ',np.sqrt(int_error_2/int_ref_2)*100, '%')
    print('Domain 3: ',np.sqrt(int_error_3/int_ref_3)*100, '%')
    print('Domain 4: ',np.sqrt(int_error_4/int_ref_4)*100, '%')
    print('Domain 5: ',np.sqrt(int_error_5/int_ref_5)*100, '%')
    print('Domain 6: ',np.sqrt(int_error_6/int_ref_6)*100, '%')
    print('Domain 7: ',np.sqrt(int_error_7/int_ref_7)*100, '%')
    print('Domain 8: ',np.sqrt(int_error_8/int_ref_8)*100, '%')
    print('Domain 9: ',np.sqrt(int_error_9/int_ref_9)*100, '%')

    print('L2 error: ',np.sqrt(int_error/int_ref)*100, '%')
    print('Exited with success!')

    output = np.array([np.sqrt(int_error/int_ref), np.sqrt(int_error_1/int_ref_1), np.sqrt(int_error_2/int_ref_2), np.sqrt(int_error_3/int_ref_3), np.sqrt(int_error_4/int_ref_4), np.sqrt(int_error_5/int_ref_5), np.sqrt(int_error_6/int_ref_6), np.sqrt(int_error_7/int_ref_7), np.sqrt(int_error_8/int_ref_8), np.sqrt(int_error_9/int_ref_9)]) 
    return output



def evaluate_error(model, params, evaluation_func, model_idxs, geoms, meshfile):
    coordinates = cal_coordinates(geoms)
    ref_values = calc_eq(meshfile, [model.mu0, model.mur], model.J0, coordinates)
    errs = cal_l2_error(model, params, geoms, meshfile)

    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
    sol_model, vmin, vmax = evaluation_func(model, params, ys, x)
    print('The min and max of the NN models is: ', vmin, vmax)

    vmin = np.amin(ref_values)
    vmax = np.amax(ref_values)
    #print('The min and max of the reference is: ', vmin, vmax)

    #vmin = 0
    #vmax = 0.05 
    error = [] 
    plt.figure()
    plt.axis('off')
    norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    m = mpl.cm.ScalarMappable(norm=norm, cmap = 'inferno')
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
        print(i+1, np.sum(np.abs(uu)))
        error.append(np.sum(error_local))
        relative_error_domain = np.sum(error_local)/np.sum(np.abs(uu))
        print('The relative error in domain ', i + 1, ' is ', relative_error_domain*100, ' %')
        plt.contourf(xx, yy, sol_model[i], norm = norm, levels = 100, cmap = 'inferno')
        #plt.contourf(xx, yy, error_local, norm = norm, levels = 100, cmap = 'inferno')
        #plt.contourf(xx, yy, uu, norm = norm, levels = 100, cmap = 'inferno')
    plt.colorbar(m, ax = plt.gca())
    # plt.show()
    plt.savefig('./ref_plot.png', bbox_inches = 'tight', pad_inches = 0)
    error_tot = np.sum(error)
    relative = error_tot/np.sum(np.abs(ref_values))
    print('The relative error amounts to ', relative*100, ' %')

    return errs


    


      



