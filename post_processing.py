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

    u1 = np.abs(u1)
    u2 = np.abs(u2)
    u3 = np.abs(u3)
    u4 = np.abs(u4)
    u5 = np.abs(u5)
    u6 = np.abs(u6)
    u7 = np.abs(u7)
    u8 = np.abs(u8)
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

    xy1 = iron_pole(ys)
    xy2 = iron_yoke(ys)
    xy3 = iron_yoke_r_mid(ys)
    xy4 = iron_yoke_r_low(ys)

    xy5 = air_1(ys)
    xy6 = air_2(ys)
    xy7 = air_3(ys)
    xy8 = current(ys)
    coors = np.concatenate((xy1, xy2, xy3, xy4, xy5, xy6, xy7, xy8))
    np.savetxt('./coordinates_new.csv', coors, delimiter = ',', comments = '')
    exit()

    vmin = np.amin(ref_values)
    vmax = np.amax(ref_values)
    
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
        xx_ref = np.reshape(local_x, (100, 100))
        yy_ref = np.reshape(local_y, (100, 100))
        uu = np.reshape(local_vals, (100, 100))
        xx = coors[i][:,0].reshape(x.shape)
        yy = coors[i][:,1].reshape(x.shape)
        if i ==2:
            plt.scatter(xx_ref.flatten(), yy_ref.flatten())
            plt.scatter(xx.flatten(), yy.flatten())
        print(np.sum(np.abs(xx_ref - xx)))
        #plt.contourf(xx,yy,uu,norm = norm, levels = 100)
    #plt.colorbar(m)
    plt.show()




