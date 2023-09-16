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


def save_weights(params, path):
    keysList = list(params.keys())
    for i in keysList:
        obj = params[i]
        if type(obj) != list:
            print('Trainable Parameter')
            obj_np = np.array(obj.tolist())
        else:
            print('Neural Network')
            print(obj[0][1][0][0][0])
            print(type(obj[0][1][0][0][0]))
            exit()
    return 0
