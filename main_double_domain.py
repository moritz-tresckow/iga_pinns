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
from mke_geo import *
from post_processing import *

# Geometry parametrizations
geoms = mke_quadrupole_geo(rnd_key)


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

        self.add_flax_network('u1', feat_domain, act_domain, load, path)
        self.add_flax_network('u5', feat_domain, act_domain, load, path)
        self.add_flax_network('u6', feat_domain, act_domain, load, path)

        # Interfaces to Air1                                
        self.add_flax_network('u15', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u16', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u12', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u56', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u67', feat_bndr, act_bndr, load, path)
        
        self.add_trainable_parameter('u156',(1,), load_p, path) 
        self.add_trainable_parameter('u1268',(1,), load_p, path) 
  
        self.interface16 = interface_function2d(1,1.0,-1.0,self.neural_networks['u16'])
        self.interface15 = interface_function2d(0,1.0,-1.0,self.neural_networks['u15'])
        self.interface12 = interface_function2d(0,-1.0,1.0,self.neural_networks['u12'])

        self.interface56 = interface_function2d(1,1.0,-1.0,self.neural_networks['u56'])
        self.interface51 = interface_function2d(0,1.0,-1.0,self.neural_networks['u15'])
        
        self.interface61 = interface_function2d(1,1.0,-1.0,self.neural_networks['u16'])
        self.interface65 = interface_function2d(0,1.0,-1.0,self.neural_networks['u56'])
        self.interface67 = interface_function2d(0,-1.0,1.0,self.neural_networks['u67'])

        
        self.mu0 = 1
        #self.mur = 2000 
        self.mur = 1
        self.J0 =  1000

        self.k1 = 0.0005
        self.k2 = 1.65/5000
        self.k3 = 0.5
        

    def get_points_MC(self, N, key):      
        points = {}

        ys = jax.random.uniform(key ,(N,2))*2-1
        Weights = jnp.ones((N,))*4/ys.shape[0]
        # ys = np.array(jax.random.uniform(self.key, (N,2)))*2-1
        # Weights = jnp.ones((N,))*4/ys.shape[0]


        points['ys1'] = ys
        points['ws1'] = Weights
        points['omega1'], points['G1'], points['K1'] = geoms[0].GetMetricTensors(ys)
       
        points['ys2'] = ys
        points['ws2'] = Weights
        points['omega2'], points['G2'], points['K2'] = geoms[1].GetMetricTensors(ys)
        
        points['ys3'] = ys
        points['ws3'] = Weights
        points['omega3'], points['G3'], points['K3'] = geoms[2].GetMetricTensors(ys)


        
        def bnd_quadrature(dom, d, end, ys):
            def eval_dL(dom, d, input_vec):
                DGys = dom._eval_omega(input_vec)
                diff = [DGys[:,0,d], DGys[:,1,d]]
                return jnp.array(diff).T

            input_vec = end*jnp.ones((N,2))
            w_quad = jnp.ones((N,))*2/ys.shape[0]
            input_vec = input_vec.at[:,d].set(ys)
            diff = eval_dL(dom, d, input_vec)
            return input_vec, diff, w_quad

        ys = -1*jnp.linspace(1,-1,N) 
        ys4, diff4, w_quad4 = bnd_quadrature(geoms[2], 0, -1, ys)
        diff4_abs = jnp.sqrt(diff4[:,0]**2 + diff4[:,1]**2)

        points['ys_bnd4'] = ys4
        points['omega_bnd4'] = diff4_abs[:,jnp.newaxis]
        points['ws_bnd4'] = w_quad4[:,jnp.newaxis]


        ys = jnp.linspace(1,-1,N) 
        ys5, diff5, w_quad5 = bnd_quadrature(geoms[2], 1, -1, ys)
        diff5_abs = jnp.sqrt(diff5[:,0]**2 + diff5[:,1]**2)

        points['ys_bnd5'] = ys5
        points['omega_bnd5'] = diff5_abs[:,jnp.newaxis]
        points['ws_bnd5'] = w_quad5[:,jnp.newaxis]


        ys = -1*jnp.linspace(1,-1,N) 
        ys6, diff6, w_quad6 = bnd_quadrature(geoms[2], 1, 1, ys)
        diff6_abs = jnp.sqrt(diff6[:,0]**2 + diff6[:,1]**2)

        points['ys_bnd6'] = ys6
        points['omega_bnd6'] = diff6_abs[:,jnp.newaxis]
        points['ws_bnd6'] = w_quad6[:,jnp.newaxis]


        ys = -1*jnp.linspace(1,-1,N) 
        ys7, diff7, w_quad7 = bnd_quadrature(geoms[0], 1, -1, ys)
        diff7_abs = jnp.sqrt(diff7[:,0]**2 + diff7[:,1]**2)

        points['ys_bnd7'] = ys7
        points['omega_bnd7'] = diff7_abs[:,jnp.newaxis]
        points['ws_bnd7'] = w_quad7[:,jnp.newaxis]


        ys = jnp.linspace(1,-1,N) 
        ys8, diff8, w_quad8 = bnd_quadrature(geoms[0], 1, 1, ys)
        diff8_abs = jnp.sqrt(diff8[:,0]**2 + diff8[:,1]**2)

        points['ys_bnd8'] = ys8
        points['omega_bnd8'] = diff8_abs[:,jnp.newaxis]
        points['ws_bnd8'] = w_quad8[:,jnp.newaxis]
        return points


   
    
    def solution1(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       3(6)
        #                                    --------  
        #                                   |        |
        # 1. Domain : Iron Pole           N |   1    |N 
        #                                   |        |
        #                                    --------    
        #                                       D
        #------------------------------------------------------------------------------#
        
        
        alpha = 2
        
        u = self.neural_networks['u1'].apply(ws['u1'],x) 

        v = ((x[...,1] + 1) * (1 - x[...,1]))[...,None]
        
        w16 = self.interface16(ws['u16'], x)

        w = w16

        output = u*v + w
        return output 




    def solution2(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                      3(6)
        #                                   +--------x156  
        #                                   |        |
        # 5. Domain : Air1                N |  2(5)  | 1
        #                                   |        |
        #                                    --------    
        #                                       D
        #------------------------------------------------------------------------------#
        
        alpha = 2
        
        u = self.neural_networks['u5'].apply(ws['u5'],x) 

        v = ((x[...,1] + 1) * (1 - x[...,1]) * (1 - x[...,0]) )[...,None]
        
        w51 = self.interface51(ws['u15'], x) * ((x[...,1] + 1) * (1 - x[...,1]))[...,None]
        
        w56 = self.interface56(ws['u56'], x) * (1 - x[...,0])[...,None]

        w156 =  ws['u156']*( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha 
        
        w = w51 + w56 + w156 
        

        output = u*v + w
        return output 
   

    def solution3(self, ws, x):

        #------------------------------------------------------------------------------#
        #                                       1 
        #                                    --------
        #                                   |        |
        # 6. Domain : Air 2               N |  6(3)  | N 
        #                                   |        |
        #                                   +--------+    
        #                                       N
        #------------------------------------------------------------------------------#
        alpha = 2
        
        u = self.neural_networks['u6'].apply(ws['u6'],x) 

        v = ((1 - x[...,1]) )[...,None]
        
        w61 = self.interface61(ws['u16'], x) 
        w = w61 
        output = u*v + w

        print('out ', output.shape)
        return output 





    def nu_model(self, grad_a):
        b2 = grad_a[...,0]**2+grad_a[...,1]**2
        return self.k1*jnp.exp(self.k2*b2)+self.k3

    def nu_model(self, b2): # Brauer Curve
        return self.k1*jnp.exp(self.k2*b2)+self.k3
    
    def loss_pde(self, ws, points):
        grad1 = src.operators.gradient(lambda x : self.solution1(ws,x))(points['ys1'])[...,0,:]
        grad3 = src.operators.gradient(lambda x : self.solution3(ws,x))(points['ys3'])[...,0,:]
        
        lpde1 = 0.5*1/(self.mu0)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad1,points['K1'],grad1), points['ws1']) 
        lpde3 = 0.5*1/self.mu0*jnp.dot(jnp.einsum('mi,mij,mj->m',grad3,points['K3'],grad3), points['ws3'])  - jnp.dot(self.J0*self.solution3(ws,points['ys3']).flatten()*points['omega3']  ,points['ws3'])
        return lpde1 + lpde3
    
    def loss_neum4(self, ws, points):
        cc = src.operators.gradient(lambda x : model.solution3(ws,x))(points['ys_bnd4'])
        out = model.solution3(ws, points['ys_bnd4'])
        cc = -1*cc[:,:,1] * out 
        cc = cc * points['omega_bnd4']
        cc = cc * points['ws_bnd4']
        val = jnp.sum(cc)
        return val 
    
    def loss_neum5(self, ws, points):
        cc = src.operators.gradient(lambda x : model.solution3(ws,x))(points['ys_bnd5'])
        out = model.solution3(ws, points['ys_bnd5'])
        cc = -1 * cc[:,:,0] * out 
        cc = cc * points['omega_bnd5']
        cc = cc * points['ws_bnd5']
        val = jnp.sum(cc)
        return val 


    def loss_neum6(self, ws, points):
        cc = src.operators.gradient(lambda x : model.solution3(ws,x))(points['ys_bnd6'])
        out = model.solution3(ws, points['ys_bnd6'])
        cc = cc[:,:,0] * out 
        cc = cc * points['omega_bnd6']
        cc = cc * points['ws_bnd6']
        val = jnp.sum(cc)
        return val 

    def loss_neum7(self, ws, points):
        cc = src.operators.gradient(lambda x : model.solution1(ws,x))(points['ys_bnd7'])
        out = model.solution1(ws, points['ys_bnd7'])
        cc = -1 * cc[:,:,0] * out 
        cc = cc * points['omega_bnd7']
        cc = cc * points['ws_bnd7']
        val = jnp.sum(cc)
        return val 


    def loss_neum8(self, ws, points):
        cc = src.operators.gradient(lambda x : model.solution1(ws,x))(points['ys_bnd8'])
        out = model.solution1(ws, points['ys_bnd8'])
        cc = cc[:,:,0] * out 
        cc = cc * points['omega_bnd8']
        cc = cc * points['ws_bnd8']
        val = jnp.sum(cc)
        return val 

    def loss(self, ws, pts):
        lpde = self.loss_pde(ws, pts)
        lpde2 = self.loss_neum4(ws, pts)
        lpde3 = self.loss_neum5(ws, pts)
        lpde4 = self.loss_neum6(ws, pts)
        lpde5 = self.loss_neum7(ws, pts)
        lpde6 = self.loss_neum8(ws, pts)
        return lpde + ( lpde2 + lpde3 + lpde4 + lpde5 + lpde6 )
    

rnd_key = jax.random.PRNGKey(1235)
model = Model(rnd_key)                  # Instantiate PINN model
w0 = model.init_unravel()               # Instantiate NN weights
weights = model.weights                 # Retrieve weights to initialize the optimizer 

#------------------------------Optimization parameters ------------------------------------#
opt_type = 'ADAM'                                                         # Optimizer name
batch_size = 50000                                                          # Number of sample points for quadrature (MC integration) 
stepsize = 0.0001                                                           # Stepsize for Optimizer aka. learning rate
n_epochs = 300                                                             # Number of optimization epochs
path_coor = './fem_ref/coordinates.csv'                                     # Path to coordinates to evaluate the NN solution
path_refs = './parameters/quad/mu_2k/ref_values.csv'                        # FEM reference solution
# meshfile = './fem_ref/fenicsx_mesh/quad_simple/quad_simple' 
meshfile = './fem_ref/fenicsx_mesh/quad_double_domain/quad_double_domain' 

opt_init, opt_update, get_params = optimizers.adamax(step_size=stepsize)    # Instantiate the optimizer
opt_state = opt_init(weights)                                               # Initialize the optimizer with the NN weights
params = get_params(opt_state)                                              # Retrieve the trainable weights for the optimizer as a dict
loss_grad = jax.jit(lambda ws, pts: (model.loss(ws, pts), jax.grad(model.loss)(ws, pts))) # JIT compile the loss function before training

key = jax.random.PRNGKey(np.random.randint(70998373))                     # Generate an PRND key to initialize the MC sampling routine
points = model.get_points_MC(batch_size, key)                               # Generate the MC samples

model.loss(params, points)
print('Source Current integral: ', jnp.dot(model.J0*points['omega3'] ,points['ws3']))
evaluate_error(model, params, evaluate_double_model, [0,1], [geoms[0], geoms[2]], meshfile)
def step(params, opt_state, key):
    points = model.get_points_MC(batch_size, key)
    loss, grads = loss_grad(params, points)                                 
    opt_state = opt_update(0, grads, opt_state)                             
    params = get_params(opt_state)                                          
    return params, opt_state, loss
#------------------------------------------------------------------------------------------#

step_compiled = jax.jit(step)                                               # JIT compile everything ...
step_compiled(params, opt_state, rnd_key)

tme = datetime.datetime.now()


#------------------------------Optimization Loop-------------------------------------------#
for k in range(n_epochs):    
    params, opt_state, loss = step_compiled(params, opt_state, jax.random.PRNGKey(np.random.randint(321323)))
    print('Epoch %d/%d - loss value %e'%(k+1, n_epochs, loss))
#------------------------------------------------------------------------------------------#


tme = datetime.datetime.now() - tme
print('Elapsed time ', tme)
save_models(params, './parameters/quad/')
print('Erfolgreich gespeichert!!')


evaluate_error(model, params, evaluate_double_model, [0,1], [geoms[0], geoms[2]], meshfile)
