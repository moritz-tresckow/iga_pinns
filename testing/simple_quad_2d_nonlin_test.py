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
from jax.config import config
from plot_scripts import plot_solution_quad_nonlin, plot_bndr_quad_nonlin
from post_processing import *
from mke_geo import sample_bnd
config.update("jax_enable_x64", True)
rnd_key = jax.random.PRNGKey(1234)

#%% Geometry parametrizations

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

   
    plot_knots = np.reshape(knots2, (knots2.shape[0]*knots.shape[1],2))

    #plt.scatter(plot_knots[:,0], plot_knots[:,1], c= 'r')
    weights = np.ones(knots2.shape[:2])
    basis1 = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basis2 = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)


    geom2 = src.geometry.PatchNURBSParam([basis1, basis2], knots2, weights, 0, 2, key)
   
    knots = np.array([ [ [0,0] , [Dc/2,0] , [Dc,0] ] , [ [ri/np.sqrt(2),ri/np.sqrt(2)] , [C[0,0],C[1,0]] , [Dc,hc] ]])
    
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
    #plt.show()
    return  geom1, geom3, geom2, geom4

#%% Instantiate geometry parametrizations

geom1, geom2, geom3, geom4 = create_geometry(rnd_key)
bnd_samples = sample_bnd(1000)
geoms = [geom1, geom2, geom3,geom4]

def save_coordinates(geoms):
    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
    outputs = []
    for i in geoms:
        out = i.__call__(ys)
        outputs.append(out)
    outputs = np.array(outputs)
    outputs = np.reshape(outputs, (outputs.shape[0]*outputs.shape[1],2))
    np.savetxt('./coordinates_simple.csv', outputs, delimiter = ',', comments = '')
    exit()

# pts,_ = geom1.importance_sampling(1000)
# 
# #plt.figure()
# plt.scatter(pts[:,0], pts[:,1], s = 1)
# 
# pts,_ = geom2.importance_sampling(1000)
# plt.scatter(pts[:,0],pts[:,1], s = 1)
# 
# pts,_ = geom3.importance_sampling(1000)
# plt.scatter(pts[:,0],pts[:,1], s = 1)
# 
# pts,_ = geom4.importance_sampling(1000)
# plt.scatter(pts[:,0],pts[:,1], s = 1)
# cs = ['r', 'b', 'k', 'y']
# [plt.scatter(geom3.__call__(i)[:,0], geom3.__call__(i)[:,1], c = j) for i,j in zip(bnd_samples, cs)]
# [plt.scatter(geom4.__call__(i)[:,0], geom4.__call__(i)[:,1], c = j) for i,j in zip(bnd_samples, cs)]
# plt.savefig('./patched_up_small.png')
# #plt.show()
# #%% Define the model
# exit()

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
        nl_bndr =8 
        load = False 
        load_p = False 
        
        feat_domain = [2, nl, nl, 1] 
        feat_bndr = [1, nl_bndr, nl_bndr, 1] 
        path = './parameters/quad_simple/'
        act_domain = nn.tanh
        act_bndr = nn.tanh

        self.add_flax_network('u1', feat_domain, act_domain, load, path)
        self.add_flax_network('u2', feat_domain, act_domain, load, path)
        self.add_flax_network('u3', feat_domain, act_domain, load, path)
        self.add_flax_network('u4', feat_domain, act_domain, load, path)
        

        self.add_flax_network('u12', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u13', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u23', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u14', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u34', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u1_0.3', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u1_0.7', feat_bndr, act_bndr, load, path)
        self.add_trainable_parameter('u123',(1,), load_p, path) # Iron - Air - Copper
        self.add_trainable_parameter('u134',(1,), load_p, path) # Iron - Iron2 - Copper
        self.add_trainable_parameter('u13_p0.33',(1,), load_p, path)
        self.add_trainable_parameter('u13_n0.33',(1,), load_p, path)
        

        # Domains: 1: Iron, 2: Air, 3: Copper, 4. Iron 2 (lower right)
        #------------------------------------------------------------------------------#
        # Iron -> Air   |   NN(x)*1/2(y+1)   
        self.interface12 = interface_function2d(1,1.0,-1.0,self.neural_networks['u12'])
        # Air -> Iron   |   NN(y)*1/2(x+1)    
        self.interface21 = interface_function2d(0,1.0,-1.0,self.neural_networks['u12'])
        # => Parametrizations turn the unit square so that x has to fit y


        # Air -> Copper   |   NN(y)* 1/2(y+1) 
        self.interface23 = interface_function2d(1,1.0,-1.0,self.neural_networks['u23'])
        # Copper -> Air   |   NN(y)* 1/2(y+1) 
        self.interface32 = interface_function2d(1,1.0,-1.0,self.neural_networks['u23'])



        # Iron -> Copper   |   NN(x)* 1/2(y+1) 
        self.interface13 = interface_function2d(0,1.0,-1.0,self.neural_networks['u13'])
        # Copper -> Iron   |   NN(x)* 1/2(y+1) 
        self.interface31 = interface_function2d(0,1.0,-1.0,self.neural_networks['u13'])

        
        # Iron -> Iron 2   |   NN(y)*-1/2(x-1) 
        self.interface14 = interface_function2d(1,-1.0,1.0,self.neural_networks['u14'])
        # Iron 2 -> Iron   |   NN(y)* 1/2(x+1) 
        self.interface41 = interface_function2d(1,1.0,-1.0,self.neural_networks['u14'])
        
        # Iron_2 -> Copper |   NN(y)*-1/2(x-1) 
        self.interface34 = interface_function2d(1,-1.0,1.0,self.neural_networks['u34'])
        # Copper -> Iron_2 |   NN(x)* 1/2(y+1) 
        self.interface43 = interface_function2d(0,1.0,-1.0,self.neural_networks['u34'])

        # Functions defining the compactly supported solution
        self.jump1 = jump_function2d(0, -0.33, self.neural_networks['u1_0.3'])
        self.jump2 = jump_function2d(0,  0.33, self.neural_networks['u1_0.7'])

        # self.mu0 = 0.001
        self.mu0 = 1
        self.mur = 2000
        # self.J0 =  1000000
        self.J0 =  1000

        self.k1 = 0.001
        self.k2 = 1.65/5000
        self.k3 = 0.5
        #num_pts = 100000
        #self.points = self.get_points_MC(1000, self.key)
        

    def get_points_MC(self, N, key):        
        points = {}

        ys = jax.random.uniform(key ,(N,2))*2-1
        Weights = jnp.ones((N,))*4/ys.shape[0]
        # ys = np.array(jax.random.uniform(self.key, (N,2)))*2-1
        # Weights = jnp.ones((N,))*4/ys.shape[0]


        points['ys1'] = ys
        points['ws1'] = Weights
        points['omega1'], points['G1'], points['K1'] = geom1.GetMetricTensors(ys)
       
        points['ys2'] = ys
        points['ws2'] = Weights
        points['omega2'], points['G2'], points['K2'] = geom2.GetMetricTensors(ys)
        
        points['ys3'] = ys
        points['ws3'] = Weights
        points['omega3'], points['G3'], points['K3'] = geom3.GetMetricTensors(ys)
       
        points['ys4'] = ys
        points['ws4'] = Weights
        points['omega4'], points['G4'], points['K4'] = geom4.GetMetricTensors(ys)

        return points


    def solution1(self, ws, x):

        #------------------------------------------------------------------------------#
        #                                       2 
        #                                   +--------x123  
        #                                   |        |
        # 1. Domain : Iron                D |   1    | 3
        #                                   |        |
        #                                   +--------x134    
        #                                       4
        #------------------------------------------------------------------------------#
        alpha = 2
        
        u = self.neural_networks['u1'].apply(ws['u1'],x) 

        v = ((x[...,1] + 1) * (1 - x[...,1]) * (x[...,0] + 1) * (1 - x[...,0]) )[...,None]
        
        w12 = self.interface12(ws['u12'], x) * ((x[...,0] + 1) * (1 - x[...,0]))[...,None]

        w13 = (self.interface13(ws['u13'], x))* ((x[...,1] + 1) * (1 - x[...,1]))[...,None]

        w14 = self.interface14(ws['u14'], x) * ((x[...,0] + 1) * (1 - x[...,0]))[...,None]

        w123 =  ws['u123']*( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha 
        
        w134 =  ws['u134']*( (x[...,0] + 1) * (1 - x[...,1] ) )[...,None]**alpha 
        
        w = w12 + w13 + w14 + w123 + w134

        return u*v+w


    def solution2(self, ws, x):

        #------------------------------------------------------------------------------#
        #                                       3 
        #                                   +--------x123  
        #                                   |        |
        # 2. Domain : Air                 N |   2    | 1
        #                                   |        |
        #                                   +--------+    
        #                                       D
        #------------------------------------------------------------------------------#
        alpha = 2
        
        u = self.neural_networks['u2'].apply(ws['u2'],x) 

        v = ((x[...,1] + 1) * (1 - x[...,1]) * (1 - x[...,0]) )[...,None]
        
        w23 = self.interface23(ws['u23'], x) * (1 - x[...,0])[...,None]

        w21 = self.interface21(ws['u12'], x) * ((x[...,1] + 1) * (1 - x[...,1]))[...,None]

        w123 =  ws['u123']*( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha 
        
        w = w21 + w23 + w123 

        return u*v+w


    
    
    def solution3(self, ws, x):

        #------------------------------------------------------------------------------#
        #                                       2 
        #                                   +--------x123  
        #                                   |        |
        # 3. Domain : Current             N |   3    | 1
        #                                   |        |
        #                                   +--------x134    
        #                                       4
        #------------------------------------------------------------------------------#
        alpha = 2
        
        u = self.neural_networks['u3'].apply(ws['u3'],x) 

        v = ((x[...,1] + 1) * (1 - x[...,1]) * (1 - x[...,0]) )[...,None]
        
        w31 = (self.interface31(ws['u13'], x))* ((x[...,1] + 1) * (1 - x[...,1]))[...,None]
        
        w32 = self.interface32(ws['u23'], x) * (1 - x[...,0])[...,None]
        
        w34 = self.interface34(ws['u34'], x) * (1 - x[...,0])[...,None]

        w123 =  ws['u123'] * ( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha 

        w134 =  ws['u134'] * ( (x[...,0] + 1) * (1 - x[...,1]) )[...,None]**alpha

        
        w = w31 + w32 + w34 + w123 + w134

        return u*v+w



    def solution4(self, ws, x):

        #------------------------------------------------------------------------------#
        #                                       1 
        #                                   +--------x134  
        #                                   |        |
        # 4. Domain : Iron2               D |   4    | 3
        #                                   |        |
        #                                   +--------+    
        #                                       N
        #------------------------------------------------------------------------------#
        alpha = 2
        
        u = self.neural_networks['u4'].apply(ws['u4'],x) 

        v = ((x[...,0] + 1) * (1 - x[...,0]) * (1 - x[...,1]) )[...,None]
        
        w41 = self.interface41(ws['u14'], x) * ((x[...,0] + 1) * (1 - x[...,0]))[...,None]
        
        w43 = self.interface43(ws['u34'], x) * (1 - x[...,1])[...,None]

        w134 =  ws['u134']*( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha 
        
        w = w41 + w43 + w134 

        return u*v+w




    def nu_model(self, grad_a):
        b2 = grad_a[...,0]**2+grad_a[...,1]**2
        return self.k1*jnp.exp(self.k2*b2)+self.k3

    def nu_model(self, b2):
        return self.k1*jnp.exp(self.k2*b2)+self.k3
    
    def loss_pde(self, ws, points):
        grad1 = src.operators.gradient(lambda x : self.solution1(ws,x))(points['ys1'])[...,0,:]
        grad2 = src.operators.gradient(lambda x : self.solution2(ws,x))(points['ys2'])[...,0,:]
        grad3 = src.operators.gradient(lambda x : self.solution3(ws,x))(points['ys3'])[...,0,:]
        grad4 = src.operators.gradient(lambda x : self.solution4(ws,x))(points['ys4'])[...,0,:]
        
        grad1x = jnp.einsum('mij,mj->mi',points['G1'],grad1)
        grad4x = jnp.einsum('mij,mj->mi',points['G4'],grad4)
        
        lpde1 = 0.5*1/(self.mu0*self.mur)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad1,points['K1'],grad1), points['ws1']) 
        lpde2 = 0.5*1/self.mu0*jnp.dot(jnp.einsum('mi,mij,mj->m',grad2,points['K2'],grad2), points['ws2'])  
        lpde3 = 0.5*1/self.mu0*jnp.dot(jnp.einsum('mi,mij,mj->m',grad3,points['K3'],grad3), points['ws3'])  - jnp.dot(self.J0*self.solution3(ws,points['ys3']).flatten()*points['omega3']  ,points['ws3'])
        lpde4 = 0.5*1/(self.mu0*self.mur)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad4,points['K4'],grad4), points['ws4']) 
        #bi1 = jnp.einsum('mi,mij,mj->m',grad1,points['K1'],grad1)
        #bi4 = jnp.einsum('mi,mij,mj->m',grad4,points['K4'],grad4)
        #lpde1 = 0.5*(self.mu0)*jnp.dot(self.nu_model(bi1)*bi1, points['ws1']) 
        #lpde2 = 0.5*jnp.dot(jnp.einsum('mi,mij,mj->m',grad2,points['K2'],grad2), points['ws2'])  
        #lpde3 = 0.5*jnp.dot(jnp.einsum('mi,mij,mj->m',grad3,points['K3'],grad3), points['ws3'])  - self.mu0*jnp.dot(self.J0*self.solution3(ws,points['ys3']).flatten()*points['omega3']  ,points['ws3'])
        #lpde4 = 0.5*(self.mu0)*jnp.dot(self.nu_model(bi4)*bi4, points['ws4'])
        return lpde1+lpde2+lpde3+lpde4

    

    def loss(self, ws, pts):
        lpde = self.loss_pde(ws, pts)
        return lpde
    


rnd_key = jax.random.PRNGKey(1235)
model = Model(rnd_key)
w0 = model.init_unravel()
weights = model.weights 

opt_type = 'ADAM'
batch_size = 20000
opt_init, opt_update, get_params = optimizers.adam(step_size=0.0001)
opt_state = opt_init(weights)

# get initial parameters
params = get_params(opt_state)
print(params['u1'])
print(params['u12'])
exit()
path_coor = './fem_ref/fenics_mesh/coordinates_simple.csv'
path_vals = './fem_ref/fenics_mesh/ref_vals_simple.csv'
path_vals = './fem_ref/fenics_mesh/quad_simple'
evaluate_error(model, params, evaluate_quad_nonlin, [0,1,2,3], geoms, path_vals)
print('Success in loading model...')
# plot_solution_quad_nonlin(model, params, [geom1, geom2, geom3, geom4])
# plot_bndr_quad_nonlin(model, weights)
# plot_solution_quad_nonlin(model, weights, [geom1, geom2, geom3, geom4])



loss_grad = jax.jit(lambda ws, pts: (model.loss(ws, pts), jax.grad(model.loss)(ws, pts)))

def step(params, opt_state, key):
    points = model.get_points_MC(batch_size, key)
    loss, grads = loss_grad(params, points)
    opt_state = opt_update(0, grads, opt_state)
    params = get_params(opt_state)
    return params, opt_state, loss

step_compiled = jax.jit(step)
step_compiled(params, opt_state, rnd_key)

n_epochs = 1000 

tme = datetime.datetime.now()
for k in range(n_epochs):    
    params, opt_state, loss = step_compiled(params, opt_state, jax.random.PRNGKey(np.random.randint(321323)))
    
    print('Epoch %d/%d - loss value %e'%(k+1, n_epochs, loss))
# update params
plot_solution_quad_nonlin(model, params, [geom1, geom2, geom3, geom4])
tme = datetime.datetime.now() - tme
print('Elapsed time ', tme)
evaluate_error(model, params, evaluate_quad_nonlin, [0,1,2,3], geoms, path_vals)
save_models(params, './parameters/quad_simple/')
