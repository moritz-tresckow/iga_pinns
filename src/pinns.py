import jax
import jax.numpy as jnp
import numpy as np
#from jax.example_libraries import stax, optimizers
import matplotlib.pyplot as plt
import datetime
import jax.scipy.optimize
import jax.flatten_util
import src
from src import models
from src.models import model_init, load_data

class PINN():
    
    def __init__(self):
        self.neural_networks = {}
        self.neural_networks_initializers = {}
        self.weights = {}
        pass
    
    def add_neural_network(self, name, ann, input_shape):
        
        self.neural_networks[name] = ann[1]
        self.neural_networks_initializers[name] = ann[0]
        self.weights[name] = ann[0](self.key, input_shape)[1]


    def add_flax_network(self, name, feat, act, load, path):
        params, model = model_init(feat, act)
        if load == True:
            try:
                params = load_data(model, path + name + '.txt')
                print('Success in loading ' + name)
            except:
                pass

        self.neural_networks[name] = model
        self.weights[name] = params


    def add_neural_network_param(self, name, ann, input_shape):
        
        self.neural_networks[name] = lambda w,x,p : ann[1](w,(x,p))
        self.neural_networks_initializers[name] = ann[0]
        self.weights[name] = ann[0](self.key, input_shape)[1]

    def add_trainable_parameter(self, name, shape, load, path):
        self.weights[name] = -1 *jax.random.normal(self.key, shape)
        if load == True:
            try:
                my_weight = np.loadtxt(path + name + '.csv', delimiter=',')
                my_weight = jnp.array(my_weight)
                self.weights[name] = my_weight
                print('Success in loading ' + name)
            except:
                pass


    def init_unravel(self):
        
        weights_vector, weights_unravel = jax.flatten_util.ravel_pytree(self.weights)
        self.weights_unravel = weights_unravel
        return weights_vector
         
    def loss(self,w):
        pass
    
    def train(self, method = 'ADAM'):
        pass

  
    def loss_handle(self, w):
        ws = self.weights_unravel(w)
        l = self.loss(ws)
        return l


    def lossgrad_handle(self, w, *args):
        ws = self.weights_unravel(w)
        
        l = self.loss(ws, *args)
        gr = jax.grad(self.loss)(ws, *args)
        
        gr,_ = jax.flatten_util.ravel_pytree(gr)
        return l, gr


        
        
