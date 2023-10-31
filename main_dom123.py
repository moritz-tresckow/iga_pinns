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

def mke_domains(key):
    a0 = [0,0]
    a1 = [0.04878132, 0]
    a2 = [0.07, 0]

    b0 = [0.03, 0.03]
    b1 = [0.04878132, 0.00536081]
    b2 = [0.07, 0.02755176]

    c0 = [0.0806066, 0.0593934]
    c1 = [0.04878132, 0.00536081]
    c3 = [0.07, 0.07]

    d0 = [0.0912132, 0.0487868]

    e0 = [0.0912132, 0]

    def mke_air1():
        knots_lower =  np.array([a0, a1, a2])
        knots_lower = knots_lower[np.newaxis, :, :]
        knots_upper =  np.array([b0, b1, b2])
        knots_upper = knots_upper[np.newaxis, :, :]

        knots = np.concatenate((knots_lower, knots_upper))
        weights = np.ones(knots.shape[:2])
        weights[1,1] = 0.7

        
        basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
        basisy = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
        air1 = src.geometry.PatchNURBSParam([basisx, basisy], knots, weights, 0, 2, key)
        return air1, knots

    def mke_air2():
        # knots_lower =  np.array([a2, b2])
        aux1 = [0.0753033,0.03286052]
        aux2 = [0.0859099,0.04347804]


        aux1x = [0.0753033,0]
        aux2x = [0.0859099,0]

        knots_lower =  np.array([e0, aux2x, aux1x, a2])
        knots_lower = knots_lower[np.newaxis, :, :]
        # knots_upper =  np.array([e0, d0])
        knots_upper =  np.array([d0,aux2, aux1, b2])
        knots_upper = knots_upper[np.newaxis, :, :]

        knots = np.concatenate((knots_lower, knots_upper))
        weights = np.ones(knots.shape[:2])

        #basisx = src.bspline.BSplineBasisJAX(np.array([-1,1]),1)
        #basisy = src.bspline.BSplineBasisJAX(np.array([-1,1]),1)
        basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
        basisy = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)


        air2 = src.geometry.PatchNURBSParam([basisx, basisy], knots, weights, 0, 2, key)
        return air2, knots


    air1, knots_air1 = mke_air1()
    air2, knots_air2 = mke_air2()
    return [air1, air2], [knots_air1, knots_air2]


def create_geometry(key, scale = 1):
    scale = scale
    Nt = 24                                                                
    lz = 40e-3                                                             
    Do = 72e-3                                                            
    Di = 51e-3                                                            
    hi = 13e-3                                                             
    bli = 3e-3                                                             
    Dc = 3.27640e-2                                                           
    hc = 7.55176e-3                                                           
    ri = 20e-3                                                           
    ra = 18e-3                                                           
    blc = hi-hc                                                           
    rm = (Dc*Dc+hc*hc-ri*ri)/(Dc*np.sqrt(2)+hc*np.sqrt(2)-2*ri)                 
    R = rm-ri
    O = np.array([rm/np.sqrt(2),rm/np.sqrt(2)])
    alpha1 = -np.pi*3/4       
    alpha2 = np.math.asin((hc-rm/np.sqrt(2))/R)
    alpha = np.abs(alpha2-alpha1)

    # Calculate the knots for the correct curvature 
    A = np.array([[O[0] - ri/np.sqrt(2), O[1] - ri/np.sqrt(2)], [O[0] - Dc, O[1] - hc]])
    b = np.array([[A[0,0]*ri/np.sqrt(2)+A[0,1]*ri/np.sqrt(2)],[A[1,0]*Dc+A[1,1]*hc]])
    C = np.linalg.solve(A,b)
    
    knots1 = np.array([[Do,Do * np.tan(np.pi/8)],[Do/np.sqrt(2),Do/np.sqrt(2)],[rm/np.sqrt(2),rm/np.sqrt(2)],[ri/np.sqrt(2),ri/np.sqrt(2)]])
    #knots2 = np.array([[Dc,hc],[Dc+blc,hi],[Di-bli,hi],[Di,hi-bli],[Di,0]])
    knots2 = np.array([[Di,hi-bli],[Di-bli,hi],[Dc+blc,hi],[Dc,hc]])
    knots3 = (knots1+knots2)/2
    knots3[-1,:] = C.flatten()
    knots = np.concatenate((knots1[None,...],knots3[None,...],knots2[None,...]),0)

    plot_knots = np.reshape(knots, (knots.shape[0]*knots.shape[1],2))

    #plt.scatter(plot_knots[:,0], plot_knots[:,1], c= 'r')
    
    
    weights = np.ones(knots.shape[:2])
    weights[1,-1] = np.sin((np.pi-alpha)/2)
    basis1 = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
    basis2 = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)

    geom1 = src.geometry.PatchNURBSParam([basis1, basis2], knots, weights, 0, 2, key)
   
    knots2 = np.array([ [ [Dc,0],[Dc+blc,0],[Di-bli,0],[Di,0] ] , [[Dc,hc],[Dc+blc,hi],[Di-bli,hi],[Di,hi-bli]] ]) 
    knots2 = knots2[:,::-1,:]

    knots_geom2 = knots2
   
    plot_knots = np.reshape(knots2, (knots2.shape[0]*knots.shape[1],2))

    #plt.scatter(plot_knots[:,0], plot_knots[:,1], c= 'r')
    weights = np.ones(knots2.shape[:2])
    basis1 = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basis2 = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)


    geom2 = src.geometry.PatchNURBSParam([basis1, basis2], knots2, weights, 0, 2, key)
   
    knots = np.array([ [ [0,0] , [Dc/2,0] , [Dc,0] ] , [ [ri/np.sqrt(2),ri/np.sqrt(2)] , [C[0,0],C[1,0]] , [Dc,hc] ]])
    
    knots_geom3 = knots
    plot_knots = np.reshape(knots, (knots.shape[0]*knots.shape[1],2))
    
    #plt.scatter(plot_knots[:,0], plot_knots[:,1], c= 'r')

    basis1 = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basis2 = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
    
    weights = np.ones(knots.shape[:2])
    weights[1,1] = np.sin((np.pi-alpha)/2)
    geom3 = src.geometry.PatchNURBSParam([basis1, basis2], knots, weights, 0, 2, key)

    knots1 = np.array([[Do,0],[Do,Do * np.tan(np.pi/8)]])
    knots2 = np.array([[Di,0],[Di,hi-bli]])
    knots3 = (knots1+knots2)/2
    knots = np.concatenate((knots1[None,...],knots3[None,...],knots2[None,...]),0)


    
    plot_knots = np.reshape(knots, (knots.shape[0]*knots.shape[1],2))

    #plt.scatter(plot_knots[:,0], plot_knots[:,1], c= 'r')
    
    
    weights = np.ones(knots.shape[:2])

    basis2 = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basis1 = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)

    geom4 = src.geometry.PatchNURBSParam([basis1, basis2], knots, weights, 0, 2, key)
    return  [geom3, geom2],[knots_geom3, knots_geom2] 

def create_geometry_alt(key, scale = 1):
    scale = scale
    Nt = 24                                                                
    lz = 40e-3                                                             
    Do = 72e-3                                                            
    Di = 51e-3                                                            
    hi = 13e-3                                                             
    bli = 3e-3                                                             
    Dc = 3.27640e-2                                                           
    hc = 7.55176e-3                                                           
    ri = 20e-3                                                           
    ra = 18e-3                                                           
    blc = hi-hc                                                           
    rm = (Dc*Dc+hc*hc-ri*ri)/(Dc*np.sqrt(2)+hc*np.sqrt(2)-2*ri)                 
    R = rm-ri
    O = np.array([rm/np.sqrt(2),rm/np.sqrt(2)])
    alpha1 = -np.pi*3/4       
    alpha2 = np.math.asin((hc-rm/np.sqrt(2))/R)
    alpha = np.abs(alpha2-alpha1)

    # Calculate the knots for the correct curvature 
    A = np.array([[O[0] - ri/np.sqrt(2), O[1] - ri/np.sqrt(2)], [O[0] - Dc, O[1] - hc]])
    b = np.array([[A[0,0]*ri/np.sqrt(2)+A[0,1]*ri/np.sqrt(2)],[A[1,0]*Dc+A[1,1]*hc]])
    C = np.linalg.solve(A,b)
    
    knots = np.array([ [ [0,0] , [Dc/2,0] , [Dc,0] ] , [ [ri/np.sqrt(2),ri/np.sqrt(2)] , [C[0,0],C[1,0]] , [Dc,hc] ]])
    print(knots)
    knots_geom3 = knots

    basis1 = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basis2 = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
    
    weights = np.ones(knots.shape[:2])
    weights[1,1] = np.sin((np.pi-alpha)/2)
    geom3 = src.geometry.PatchNURBSParam([basis1, basis2], knots, weights, 0, 2, key)
    
    # knots_upper = [[Dc,0] ,[Di,0] ]
    #knots_upper = [[Di,0] ,[Dc,0] ]
    # knots_lower = [[Dc,hc], [Di,hi+bli]]
    #knots_lower = [[Di,hi+bli], [Dc,hc]]
  
    b2 = 0.05
    hx = 0.009312
    hy = 0.003294  

    knots_upper = [[b2 + 2*hx,0] ,[Dc,0] ]
    knots_lower = [[b2 + 2*hx, b2 - 2*hy], [Dc,hc]]
    knots2 = np.array([knots_upper , knots_lower ]) 
    #knots2 = knots2[:,::-1,:]

    knots_geom2 = knots2
   
    #plt.scatter(plot_knots[:,0], plot_knots[:,1], c= 'r')
    weights = np.ones(knots2.shape[:2])
    basis1 = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basis2 = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)

    geom2 = src.geometry.PatchNURBSParam([basis1, basis2], knots2, weights, 0, 2, key)



    knots_lower = np.array([ [ri/np.sqrt(2),ri/np.sqrt(2)] , [C[0,0],C[1,0]] , [Dc,hc] ])
    knots_upper = np.array([ [b2, b2], [b2 + hx, b2 - hy] , [b2 + 2*hx, b2 - 2*hy] ])
    knots_geom1 = np.array([knots_upper , knots_lower ]) 
    
    basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basisy = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
    
    weights = np.ones(knots.shape[:2])
    weights[1,1] = np.sin((np.pi-alpha)/2)
    geom1 = src.geometry.PatchNURBSParam([basisx, basisy], knots_geom1, weights, 0, 2, key)
    return  [geom3, geom2, geom1] ,[knots_geom3, knots_geom2, knots_geom1] 


def mke_geo(rnd_key):  
    geoms, knots = create_geometry_alt(rnd_key)
    meshfile = './fem_ref/fenicsx_mesh/quad_dom123/quad_dom123' 
    return geoms, knots, meshfile

geoms, knots, meshfile = mke_geo(rnd_key)
bnd_samples = sample_bnd(1000)

pts = [i.importance_sampling(100000)[0] for i in geoms]
[plt.scatter(i[:,0], i[:,1], s=1) for i in pts]
knots_pts = [np.reshape(i, (i.shape[0]*i.shape[1],i.shape[2])) for i in knots]

cs = ['r', 'b', 'k', 'y']
l = 0
[plt.scatter(geoms[l].__call__(i)[:,0], geoms[l].__call__(i)[:,1], c = j, s=5) for i,j in zip(bnd_samples, cs)]
[plt.scatter(i[:,0], i[:,1], c = 'r') for i in knots_pts]
plt.savefig('dom123.png')

def interface_function2d(nd, endpositive, endzero, nn):
    faux = lambda x: ((x-endzero)**1/(endpositive-endzero)**1)
    if nd == 0: # NN(y)*(x-endzero)/(endpositive - endzero)
        fret = lambda ws, x: (nn.apply(ws, x[...,1][...,None]).flatten()*faux(x[...,0]))[...,None]
    else: # NN(x)*(y-endzero)/(endpositive - endzero)
        fret = lambda ws, x: (nn.apply(ws, x[...,0][...,None]).flatten()*faux(x[...,1]))[...,None]
    return fret

class Model(src.PINN):
    def __init__(self, rand_key):
        super().__init__()
        self.key = rand_key

        nl = 16 
        nl_bndr = 8 
        load = True 
        load_p = True 
        path = './parameters/quad/'

        feat_domain = [2, nl, nl, nl, 1] 
        act_domain = nn.tanh
        feat_bndr = [1, nl_bndr, nl_bndr, nl_bndr, 1] 
        act_bndr = nn.tanh

        self.add_flax_network('u1', feat_domain, act_domain, load, path)
        self.add_flax_network('u2', feat_domain, act_domain, load, path)
        self.add_flax_network('u3', feat_domain, act_domain, load, path)
        self.add_flax_network('u12', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u13', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u23', feat_bndr, act_bndr, load, path)

        self.interface12 = interface_function2d(1,1.0,-1.0,self.neural_networks['u12'])
        self.interface21 = interface_function2d(1,1.0,-1.0,self.neural_networks['u12'])

        self.interface13 = interface_function2d(0,1.0,-1.0,self.neural_networks['u13'])
        self.interface31 = interface_function2d(0,1.0,-1.0,self.neural_networks['u13'])

        self.interface23 = interface_function2d(0,1.0,-1.0,self.neural_networks['u23'])
        self.interface32 = interface_function2d(1,1.0,-1.0,self.neural_networks['u23'])

        self.add_trainable_parameter('u123',(1,), load_p, path) 
        
        self.mu0 = 1
        self.mur = 2000 
        self.J0 =  1000
        

    def get_points_MC(self, N, key):      
        points = {}

        ys = jax.random.uniform(key ,(N,2))*2-1
        Weights = jnp.ones((N,))*4/ys.shape[0]

        points['ys1'] = ys
        points['ws1'] = Weights
        points['omega1'], points['G1'], points['K1'] = geoms[0].GetMetricTensors(ys)
       
        points['ys2'] = ys
        points['ws2'] = Weights
        points['omega2'], points['G2'], points['K2'] = geoms[1].GetMetricTensors(ys)

        points['ys3'] = ys
        points['ws3'] = Weights
        points['omega3'], points['G3'], points['K3'] = geoms[2].GetMetricTensors(ys)
        
        return points


    
    def solution1(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       2 
        #                                    --------x123
        #                                   |        |
        # 1. Domain : Air gap             N |   1    | 3 
        #                                   |        |
        #                                    --------    
        #                                       D
        #------------------------------------------------------------------------------#
        
        alpha = 2 
        
        u = self.neural_networks['u1'].apply(ws['u1'],x) 

        v = ((1 - x[...,1]) *  (x[...,1] + 1) * (1 - x[...,0]))[...,None]
        
        w12 = self.interface12(ws['u12'], x) *  (1 - x[...,0])[...,None]         
        w13 = self.interface13(ws['u13'], x) * ((1 - x[...,1]) *  (x[...,1] + 1))[...,None]

        w123  = ws['u123'] *( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha

        w = w12 + w13 + w123

        output = u*v + w
        return output 


    def solution2(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       1 
        #                                    --------x123  
        #                                   |        |
        # 2. Domain : Air2                N |   2    | 3
        #                                   |        |
        #                                    --------    
        #                                       N
        #------------------------------------------------------------------------------#
        
        alpha = 2 
        
        u = self.neural_networks['u2'].apply(ws['u2'],x) 

        v = ((1 - x[...,1]) * (1 - x[...,0]))[...,None]
        
        w21 = self.interface21(ws['u12'], x) * (1 - x[...,0])[...,None]         
        w23 = self.interface23(ws['u23'], x) * (1 - x[...,1])[...,None]         
        
        w123  = ws['u123'] *( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha
        
        w = w21 + w23 + w123

        output = u*v + w
        return output 
   


    def solution3(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       2 
        #                                    --------x123  
        #                                   |        |
        # 3. Domain : Iron Pole           N |   3    | 1
        #                                   |        |
        #                                    --------    
        #                                       D
        #------------------------------------------------------------------------------#
        alpha = 2 
        
        u = self.neural_networks['u3'].apply(ws['u3'],x) 

        v = ((1 - x[...,0]) * (1 - x[...,1]) * (x[...,1] + 1))[...,None]
        
        w31 = self.interface31(ws['u13'], x) * ((1 - x[...,1]) *  (x[...,1] + 1))[...,None]
        w32 = self.interface32(ws['u23'], x) *  (1 - x[...,0]) [...,None]
        
        w123  = ws['u123'] *( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha
        
        w = w31 + w32 + w123

        output = u*v + w
        return output 
    
    def loss_pde(self, ws, points):
        grad1 = src.operators.gradient(lambda x : self.solution1(ws,x))(points['ys1'])[...,0,:]
        grad2 = src.operators.gradient(lambda x : self.solution2(ws,x))(points['ys2'])[...,0,:]
        grad3 = src.operators.gradient(lambda x : self.solution3(ws,x))(points['ys3'])[...,0,:]
        
        lpde1 = 0.5*1/(self.mu0)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad1,points['K1'],grad1), points['ws1'])

        lpde2 = 0.5*1/self.mu0*jnp.dot(jnp.einsum('mi,mij,mj->m',grad2,points['K2'],grad2), points['ws2'])  - jnp.dot(self.J0*self.solution2(ws,points['ys2']).flatten()*points['omega2'], points['ws2'])

        lpde3 = 0.5*1/(self.mu0*self.mur)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad3,points['K3'],grad3), points['ws3']) 
        return lpde1 + lpde2 + lpde3


    def loss(self, ws, pts):
        lpde = self.loss_pde(ws, pts)
        return lpde 
    

rnd_key = jax.random.PRNGKey(1235)
model = Model(rnd_key)                  # Instantiate PINN model
w0 = model.init_unravel()               # Instantiate NN weights
weights = model.weights                 # Retrieve weights to initialize the optimizer 

#------------------------------Optimization parameters ------------------------------------#
opt_type = 'ADAM'                                                         # Optimizer name
batch_size = 50000                                                          # Number of sample points for quadrature (MC integration) 
stepsize = 0.0001                                                           # Stepsize for Optimizer aka. learning rate
n_epochs = 1000                                                             # Number of optimization epochs
path_coor = './fem_ref/coordinates.csv'                                     # Path to coordinates to evaluate the NN solution
path_refs = './parameters/quad/mu_2k/ref_values.csv'                        # FEM reference solution
# meshfile = './fem_ref/fenicsx_mesh/quad_simple/quad_simple' 

opt_init, opt_update, get_params = optimizers.adamax(step_size=stepsize)    # Instantiate the optimizer
opt_state = opt_init(weights)                                               # Initialize the optimizer with the NN weights
params = get_params(opt_state)                                              # Retrieve the trainable weights for the optimizer as a dict
loss_grad = jax.jit(lambda ws, pts: (model.loss(ws, pts), jax.grad(model.loss)(ws, pts))) # JIT compile the loss function before training

key = jax.random.PRNGKey(1223435)
points = model.get_points_MC(batch_size, key)                               # Generate the MC samples

model.loss_pde(params, points)
evaluate_error(model, params, evaluate_triple_model, [0,1,2], geoms, meshfile)
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


evaluate_error(model, params, evaluate_triple_model, [0,1,2], geoms, meshfile)
