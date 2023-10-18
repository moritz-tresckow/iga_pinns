import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers
import matplotlib.pyplot as plt
import src 
from src import models
from src.models import *

import datetime
import jax.scipy.optimize
import jax.flatten_util
import scipy
import scipy.optimize
#from helpers import write_data
from jax.config import config
config.update("jax_enable_x64", True)
rnd_key = jax.random.PRNGKey(1234)
from mke_geo import create_geometry, plot_solution, plot_single_domain, plot_bndr
from post_processing import evaluate_models, evaluate_error, evaluate_air



# Geometry parametrizations
iron_pole, iron_yoke, iron_yoke_r_mid, iron_yoke_r_low, air_1, air_2, air_3, current  = create_geometry(rnd_key)
#air_1, air_2 = mke_merged_patch(rnd_key)

def interface_function2d(nd, endpositive, endzero, nn):
    # Interface function whether the interface is in x or in y direction
    # Connect the correct basis functions
    # NN is defined on the boundary so only takes in 1 dimensional inputs

    faux = lambda x: ((x-endzero)**1/(endpositive-endzero)**1)
    if nd == 0: # NN(y)*(x-endzero)/(endpositive - endzero)
        fret = lambda ws, x: (nn.apply(ws, x[...,1][...,None]).flatten()*faux(x[...,0]))[...,None]
    else: # NN(x)*(y-endzero)/(endpositive - endzero)
        fret = lambda ws, x: (nn.apply(ws, x[...,0][...,None]).flatten()*faux(x[...,1]))[...,None]
    return fret


def interface_function2d_inv(nd, endpositive, endzero, nn):
    # Interface function whether the interface is in x or in y direction
    # Connect the correct basis functions
    # NN is defined on the boundary so only takes in 1 dimensional inputs

    faux = lambda x: ((x-endzero)**1/(endpositive-endzero)**1)
    if nd == 0: # NN(y)*(x-endzero)/(endpositive - endzero)
        fret = lambda ws, x: (nn.apply(ws, -1 * x[...,1][...,None]).flatten()*faux(x[...,0]))[...,None]
    else: # NN(x)*(y-endzero)/(endpositive - endzero)
        fret = lambda ws, x: (nn.apply(ws, -1 * x[...,0][...,None]).flatten()*faux(x[...,1]))[...,None]
    return fret



def jump_function2d(nd, pos_y, nn):
    # Function compactly supported on the patch
    faux = lambda x: jnp.exp(-4.0*jnp.abs(x-pos_y))
    if nd == 1:
        fret = lambda ws, x: (nn.apply(ws, x[...,1][...,None]).flatten()*faux(x[...,0]))[...,None]
    else: # fret(x,y) = NN(x)*exp(-4*|y-y_pos|)
        fret = lambda ws, x: (nn.apply(ws, x[...,0][...,None]).flatten()*faux(x[...,1]))[...,None]
    return fret


def ExpHat(x, scale = 0.1):
    # Interface function implementing continuity across patches
    return jnp.exp(-jnp.abs(x)/scale)

class Model(src.PINN):
    def __init__(self, rand_key):
        super().__init__()
        self.key = rand_key

        nl = 16 
        nl_bndr = 5 
        load = True 
        load_p = True 
        path = './parameters/quad/'

        feat_domain = [2, nl, nl, nl, 1] 
        act_domain = nn.tanh
        feat_bndr = [1, nl_bndr, nl_bndr, nl_bndr, 1] 
        act_bndr = nn.tanh

        # 5 Air1                                                
        self.add_flax_network('u5', feat_domain, act_domain, load, path)
        # 7 Air3                                                  
        self.add_flax_network('u7', feat_domain, act_domain, load, path)
        # 8 Current                                                
        self.add_flax_network('u8', feat_domain, act_domain, load, path)

        # Interfaces to Air1                                
        self.add_flax_network('u57', feat_bndr, act_bndr, load, path)
                                                             
        # Interfaces to Air3                                    
        self.add_flax_network('u78', feat_bndr, act_bndr, load, path)

        # self.add_trainable_parameter('u156',(1,), load_p, path) 
        self.add_trainable_parameter('u578',(1,), load_p, path) 

        

        # Domains: 1: PoleTip, 2: IronYoke, 3: IronYoke Right Middle, 4. IronYoke Right Lower, 
        #          5. Air1,    6. Air2,     7. Air3,                  8. Current
        #------------------------------------------------------------------------------#
        # Air1 -> Air3   |   NN(x)* 1/2(y+1)   => (1, 1, -1) 
        self.interface57 = interface_function2d(1, 1.0, -1.0,self.neural_networks['u57'])
        # Air3 -> Air1   |   NN(x)* 1/2(y+1)  => (1, 1, -1) 
        self.interface75 = interface_function2d_inv(1, 1.0, -1.0,self.neural_networks['u57'])
        

        #------------------------------------------------------------------------------#
        # Air3 -> Current   |   NN(y)* 1/2(x+1)   => (0, 1, -1) 
        self.interface78 = interface_function2d(0, 1.0, -1.0,self.neural_networks['u78'])
        # Current -> Air3   |   NN(y)* 1/2(x+1)   => (0, 1, -1) 
        self.interface87 = interface_function2d(0, 1.0, -1.0,self.neural_networks['u78'])
        
        
        self.mu0 = 1
        self.mur = 1e11
        self.J0 =  1000

        self.k1 = 0.0005
        self.k2 = 1.65/5000
        self.k3 = 0.5
        

    def get_points_MC(self, N, key):      
        points = {}

        ys = jax.random.uniform(key ,(N,2))*2-1
        Weights = jnp.ones((N,))*4/ys.shape[0]


        points['ys5'] = ys
        points['ws5'] = Weights
        points['omega5'], points['G5'], points['K5'] = air_1.GetMetricTensors(ys)
        
        points['ys7'] = ys
        points['ws7'] = Weights
        points['omega7'], points['G7'], points['K7'] = air_3.GetMetricTensors(ys)
       
        points['ys8'] = ys
        points['ws8'] = Weights
        points['omega8'], points['G8'], points['K8'] = current.GetMetricTensors(ys)
        
        return points

    def solution5(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       7
        #                             (578) x--------  
        #                                   |        |
        # 5. Domain : Air1                N |   5    | 1
        #                                   |        |
        #                                    --------    
        #                                       D
        #------------------------------------------------------------------------------#
        alpha = 2

        # NN defined on the Air1 domain
        u = self.neural_networks['u5'].apply(ws['u5'],x)
        
        # Ansatz Function: v(x,y) = (1-y)*(1+y) -> (x+1)
        #------------------------------------------------------------------------------#
        v = ((1 - x[...,1]) * (x[...,1] + 1))[...,None]
        

        w57 = self.interface56(ws['u57'],x) * (1 - x[...,0])[...,None]
        #------------------------------------------------------------------------------#
        # w57 = NN_{57}(x) * 1/2(y+1) * (1-x)                   |   

        # Function coinciding on the multiple subdomains
        #------------------------------------------------------------------------------#
        w578 = ws['u578']*( (1 - x[...,0]) * ( x[...,1] + 1) )[...,None]**alpha
        #------------------------------------------------------------------------------#
        w =  w57 + w578
        return u * v + w


    def solution7(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       4
        #                                    --------  
        #                                   |        |
        # 7. Domain : Air3                N |   7    | 8
        #                                   |        |
        #                                    --------x578    
        #                                       5
        #------------------------------------------------------------------------------#
        alpha = 2
        
        # NN defined in the Air3 domain
        u = self.neural_networks['u7'].apply(ws['u7'],x)

        # Ansatz Function: v(x,y) = (x+1)*(1-y)*(y+1) -> (1-x) missing due to Neumann bc
        #------------------------------------------------------------------------------#
        v = ((1 - x[...,0] ) * (x[...,1] + 1))[...,None]
        
        # Interface functions for the Air3 domain 
        #------------------------------------------------------------------------------#
        w78 = (self.interface78(ws['u78'],x)) * ((1 - x[...,1]))[...,None]
        #------------------------------------------------------------------------------#
        # w78 = (NN_{78}(y) * (1/2(x+1)) * (1-y)      |

        # Function coinciding on multiple subdomains
        #------------------------------------------------------------------------------#
        w578  = ws['u578']  *  ( (x[...,0] + 1) * (1 - x[...,1]) )[...,None]**alpha

        #------------------------------------------------------------------------------#
        # w578  = u_{578}  * ((x+1)*(1-y))^alpha    |

        w = w78 + w578
        return u * v + w



    def solution8(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       7a
        #                                    --------  
        #                                   |        |
        # 8. Domain : Current             2 |   8    | 7b
        #                                   |        |
        #                                    --------x578    
        #                                       3
        #------------------------------------------------------------------------------#
        alpha = 2

        # NN defined in the coil domain
        u = self.neural_networks['u8'].apply(ws['u8'],x) 
        
        # Ansatz Function: v(x,y) = (1-x)*(1-y)
        #------------------------------------------------------------------------------#
        v = ((1 - x[...,0]) * (1 - x[...,1]) )[...,None]



        w87 = (self.interface87(ws['u78'],x)) * ((1 - x[...,1]))[...,None]
        w87a = ws['u87'].apply(ws, (0.5 * (x[...,0] - 1)[...,None])).flatten()*(1/2)*(x[...,1] + 1)[...,None]
        w87b = ws['u87'].apply(ws, (-0.5 * (x[...,0] - 1)[...,None])).flatten()*(1/2)*(x[...,0] + 1)[...,None]
        


                              #------------------------------------------------------------------------------#
        # w87 = (                                                          |  
        #           NN_{78}(y)* 1/2(x+1)*(1-y)(y+1)                        |  
        
        # Function coinciding on multiple subdomains
        #------------------------------------------------------------------------------#
                              # w578  = ws['u578'] * ( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha 
        w =  w87a + w87b #+ w238# w82 + w83     + w3478+ w1268
        return u * v + w
        

    def nu_model(self, grad_a):
        b2 = grad_a[...,0]**2+grad_a[...,1]**2
        return self.k1*jnp.exp(self.k2*b2)+self.k3

    def nu_model(self, b2): # Brauer Curve
        return self.k1*jnp.exp(self.k2*b2)+self.k3
    
    def loss_pde(self, ws, points):
        # Calculate the spatial gradients grad(u) = (u_x, u_y) with at the quadrature points

        grad5 = src.operators.gradient(lambda x : self.solution5(ws,x))(points['ys5'])[...,0,:]
        grad7 = src.operators.gradient(lambda x : self.solution7(ws,x))(points['ys7'])[...,0,:]
        grad8 = src.operators.gradient(lambda x : self.solution8(ws,x))(points['ys8'])[...,0,:]
        
        #---------------------------------Air + Excitation------------------------------------------------------------------# 
        lpde5 = 0.5 * 1/self.mu0 * jnp.dot(jnp.einsum('mi,mij,mj->m',grad5,points['K5'],grad5), points['ws5'])  
        lpde7 = 0.5 * 1/self.mu0 * jnp.dot(jnp.einsum('mi,mij,mj->m',grad7,points['K7'],grad7), points['ws7'])  
        lpde8 = 0.5 * 1/self.mu0 * jnp.dot(jnp.einsum('mi,mij,mj->m',grad8,points['K8'],grad8), points['ws8'])  \
                        - jnp.dot(self.J0*self.solution8(ws,points['ys8']).flatten()*points['omega8'] , points['ws8'])
        #-------------------------------------------------------------------------------------------------------------------#


        # Sum up losses from the individual subdomains
        lpde_air  = lpde5+lpde7+lpde8
        return lpde_air #+ lpde_iron     
    

    def loss_neum(self, ws, points):
        cc = src.operators.gradient(lambda x : model.solution7(ws,x))(points['ys_bnd7'])
        out = model.solution7(ws, points['ys_bnd7'])
        cc = cc[:,:,1] * out * points['omega_bnd7']
        cc= cc* points['ws_bnd7']
        val = jnp.sum(cc)
        return val 

    def loss(self, ws, pts):
        lpde = self.loss_pde(ws, pts)
        return lpde 
    

rnd_key = jax.random.PRNGKey(1235)
model = Model(rnd_key)                  # Instantiate PINN model
w0 = model.init_unravel()               # Instantiate NN weights
weights = model.weights                 # Retrieve weights to initialize the optimizer 

#------------------------------Optimization parameters ------------------------------------#
opt_type = 'ADAMax'                                                         # Optimizer name
batch_size = 10000                                                          # Number of sample points for quadrature (MC integration) 
stepsize = 0.001                                                           # Stepsize for Optimizer aka. learning rate
n_epochs = 250                                                             # Number of optimization epochs
path_coor = './fem_ref/coordinates.csv'                                     # Path to coordinates to evaluate the NN solution
path_refs = './parameters/quad/mu_2k/ref_values.csv'                        # FEM reference solution

opt_init, opt_update, get_params = optimizers.adamax(step_size=stepsize)    # Instantiate the optimizer
opt_state = opt_init(weights)                                               # Initialize the optimizer with the NN weights
params = get_params(opt_state)                                              # Retrieve the trainable weights for the optimizer as a dict

ys = np.linspace(-1,1,100)
ys = ys[:,np.newaxis]
one_vec = np.ones_like(ys)
ys_right = np.concatenate((one_vec, ys), axis = 1)
ys_top = np.concatenate((ys, one_vec), axis = 1)
print('start calc')
sol = model.solution8(params, ys_top)
print(sol)
exit()



evaluate_error(model, params, evaluate_air,[4,5,6,7], path_coor, path_refs)        # Evaluate the model error before training
loss_grad = jax.jit(lambda ws, pts: (model.loss(ws, pts), jax.grad(model.loss)(ws, pts))) # JIT compile the loss function before training

key = jax.random.PRNGKey(np.random.randint(777623))                     # Generate an PRND key to initialize the MC sampling routine
points = model.get_points_MC(batch_size, key)                               # Generate the MC samples

#------------------------------Optimization Step-------------------------------------------#
def step(params, opt_state, key):
    # points = model.get_points_MC(batch_size, key)
    loss, grads = loss_grad(params, points)                                 # Calculate the loss with respect to the MC samples
    opt_state = opt_update(0, grads, opt_state)                             # Update the optimizer
    params = get_params(opt_state)                                          # Retrieve the new NN parameters
    return params, opt_state, loss
#------------------------------------------------------------------------------------------#

step_compiled = jax.jit(step)                                               # JIT compile everything ...
step_compiled(params, opt_state, rnd_key)

tme = datetime.datetime.now()


#------------------------------Optimization Loop-------------------------------------------#
for k in range(n_epochs):    
    params, opt_state, loss = step_compiled(params, opt_state, key)
    print('Epoch %d/%d - loss value %e'%(k+1, n_epochs, loss))
#------------------------------------------------------------------------------------------#


tme = datetime.datetime.now() - tme
print('Elapsed time ', tme)
save_models(params, './parameters/quad/')
print('Erfolgreich gespeichert!!')
evaluate_error(model, params, evaluate_air,[4,5,6,7], path_coor, path_refs)
