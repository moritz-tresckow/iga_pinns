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

def create_geo_II(key):
    b2 = 0.05
    hx = 0.009312
    hy = 0.003294  
    
    startx = b2 + 2*hx
    starty = b2 - 2*hy    
    dd = 0.01

    l1 = [b2, b2]
    l2 = [b2 + 1.5*dd, b2 + 1.5*dd]
    l3 = [b2 + 4*dd, b2 + 4*dd]
    l4 = [b2 + 7.5*dd, b2]
    knots_lower = np.array([l4, l3, l2 ,l1])

    u1 = [startx, starty]
    u2 = [startx + 0.75*dd, starty + 0.75*dd]
    u3 = [startx + (0.75 + 1.3876)*dd, starty + 0.75*dd]
    u4 = [startx + (1.5 + 1.3876)*dd, starty]

    knots_upper = np.array([u4, u3, u2, u1])
    
    knots_middle = 0.5*(knots_upper + knots_lower)

    knots4 = np.concatenate((knots_lower[None,...],knots_middle[None,...],knots_upper[None,...]),0)
    weights = np.ones(knots4.shape[:2])
    
    basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,3),1)
    basisy = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)
    geom4 = src.geometry.PatchNURBSParam([basisx, basisy], knots4, weights, 0, 2, key)



    knots_upper =  np.array([u4, u3, u2 ,u1])

    l1 = [startx + 0.5*dd               , starty - dd]
    l2 = [startx + (0.75 + 1.3876/2)*dd , starty - dd]
    l3 = [startx + (0.75 + 1.3876)*dd   , starty - dd]
    l4 = [startx + (1.5 + 1.3876)*dd    , starty - dd]
    knots_lower = np.array([l4, l3, l2 ,l1])

    knots5 = np.concatenate((knots_lower[None,...], knots_upper[None,...]),0)
    weights = np.ones(knots5.shape[:2])
    
    basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basisy = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)
    geom5 = src.geometry.PatchNURBSParam([basisx, basisy], knots5, weights, 0, 2, key)
    u1 = [startx + 0.5*dd               , starty - dd]
    u2 = [startx + (0.75 + 1.3876/2)*dd , starty - dd]
    u3 = [startx + (0.75 + 1.3876)*dd   , starty - dd]
    u4 = [startx + (1.5 + 1.3876)*dd    , starty - dd]

    

    l1 = [startx + 0.5*dd               , 0 ]
    l2 = [startx + (0.75 + 1.3876/2)*dd , 0 ]
    l3 = [startx + (0.75 + 1.3876)*dd   , 0 ]
    l4 = [startx + (1.5 + 1.3876)*dd    , 0 ]
    
    #knots51 = np.array([l1, u1])
    #knots52 = np.array([l2, u2])
    #knots53 = np.array([l3, u3])
    #knots54 = np.array([l4, u4])
    
    knots_lower = np.array([l4, l3, l2, l1])
    knots_upper = np.array([u4, u3, u2, u1])

    knots6 = np.concatenate((knots_upper[None,...], knots_lower[None,...]),0)
    #knots6 = np.concatenate((knots54[None,...], knots53[None,...], knots52[None,...], knots51[None,...]),0)
    weights = np.ones(knots6.shape[:2])
    
    basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basisy = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)
    geom6 = src.geometry.PatchNURBSParam([basisx, basisy], knots6, weights, 0, 2, key)

    u1 = [startx, starty]
    u2 = [startx + 0.5*dd, starty - dd]

    l1 = [startx, 0]
    l2 = [startx + 0.5*dd, 0]

    knots_upper = np.array([l1, u1])
    knots_lower = np.array([l2 ,u2])

    knots7 = np.concatenate((knots_lower[None,...], knots_upper[None,...]),0)
    weights = np.ones(knots7.shape[:2])
    
    basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basisy = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    geom7 = src.geometry.PatchNURBSParam([basisx, basisy], knots7, weights, 0, 2, key)

    return [geom4, geom5, geom6, geom7], [knots4, knots5, knots6, knots7]


def create_geo_III(key):
    b2 = 0.05
    hx = 0.009312
    hy = 0.003294  
    
    startx = b2 + 2*hx
    starty = b2 - 2*hy    
    dd = 0.01

    u1 = [startx + (1.5 + 1.3876)*dd, starty]
    u2 = [b2 + 7.5*dd, b2]

    l1 = [startx + (1.5 + 1.3876)*dd, starty-dd]
    l2 = [b2 + 7.5*dd, starty - dd]
    knots_upper = np.array([u1, u2])
    knots_lower = np.array([l1, l2])

    knots8 = np.concatenate((knots_lower[None,...], knots_upper[None,...]),0)
    weights = np.ones(knots8.shape[:2])
    
    basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basisy = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    geom8 = src.geometry.PatchNURBSParam([basisx, basisy], knots8, weights, 0, 2, key)
    
    u1 = [startx + (1.5 + 1.3876)*dd, starty-dd]
    u2 = [b2 + 7.5*dd, starty - dd]
    l1 = [startx + (1.5 + 1.3876)*dd, 0]
    l2 = [b2 + 7.5*dd, 0]
    knots_upper = np.array([u2, u1])
    knots_lower = np.array([l2, l1])

    knots9 = np.concatenate((knots_lower[None,...], knots_upper[None,...]),0)
    weights = np.ones(knots9.shape[:2])
    
    basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    basisy = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
    geom9 = src.geometry.PatchNURBSParam([basisx, basisy], knots9, weights, 0, 2, key)
    return [geom8, geom9], [knots8, knots9]



def mke_geo(rnd_key):  
    geoms, knots = create_geometry_alt(rnd_key)
    meshfile = './fem_ref/fenicsx_mesh/quad_doms/quad_doms_wider_yoke' 
    return geoms, knots, meshfile

geoms, knots, meshfile = mke_geo(rnd_key)
geom4, knots4 = create_geo_II(rnd_key)
geom8, knots8 = create_geo_III(rnd_key)
geoms = geoms + geom4 + geom8
knots = knots + knots4 + knots8

bnd_samples = sample_bnd(1000)

pts = [i.importance_sampling(1000)[0] for i in geoms]
c = ['b', 'b', 'k', 'k', 'y', 'b', 'b', 'k', 'k']
[plt.scatter(i[:,0], i[:,1], c = j, s = 2) for i,j in zip(pts,c)]
knots_pts = [np.reshape(i, (i.shape[0]*i.shape[1],i.shape[2])) for i in knots]

cs = ['r', 'b', 'g', 'y']
l = 3
[plt.scatter(geoms[l].__call__(i)[:,0], geoms[l].__call__(i)[:,1], c = j, s=5) for i,j in zip(bnd_samples, cs)]
[plt.scatter(i[:,0], i[:,1], c = 'r') for i in knots_pts]
plt.savefig('doms.png')


def interface_function2d(nd, endpositive, endzero, nn):
    faux = lambda x: ((x-endzero)**1/(endpositive-endzero)**1)
    if nd == 0: # NN(y)*(x-endzero)/(endpositive - endzero)
        fret = lambda ws, x: (nn.apply(ws, x[...,1][...,None]).flatten()*faux(x[...,0]))[...,None]
    else: # NN(x)*(y-endzero)/(endpositive - endzero)
        fret = lambda ws, x: (nn.apply(ws, x[...,0][...,None]).flatten()*faux(x[...,1]))[...,None]
    return fret


def interface_function2d_inv(nd, endpositive, endzero, nn):
    faux = lambda x: ((x-endzero)**1/(endpositive-endzero)**1)
    if nd == 0: # NN(y)*(x-endzero)/(endpositive - endzero)
        fret = lambda ws, x: (nn.apply(ws, -1 * x[...,1][...,None]).flatten()*faux(x[...,0]))[...,None]
    else: # NN(x)*(y-endzero)/(endpositive - endzero)
        fret = lambda ws, x: (nn.apply(ws, -1 * x[...,0][...,None]).flatten()*faux(x[...,1]))[...,None]
    return fret


class Model(src.PINN):
    def __init__(self, rand_key):
        super().__init__()
        self.key = rand_key

        nl = 8 
        nll = 16
        nl_bndr = 5 
        load = True 
        load_p =True 
        path = './parameters/quad/'

        feat_domain = [2, nl, nl, nl, 1] 
        act_domain = nn.tanh

        feat_domain_more = [2, nll, nll, nll, 1] 

        feat_bndr = [1, nl_bndr, nl_bndr, nl_bndr, 1] 
        act_bndr = nn.tanh

        self.add_flax_network('u1', feat_domain, act_domain, load, path)
        self.add_flax_network('u2', feat_domain, act_domain, load, path)
        self.add_flax_network('u3', feat_domain_more, act_domain, load, path)
        self.add_flax_network('u4', feat_domain_more, act_domain, load, path)
        self.add_flax_network('u5', feat_domain, act_domain, load, path)
        self.add_flax_network('u6', feat_domain, act_domain, load, path)
        self.add_flax_network('u7', feat_domain, act_domain, load, path)
        self.add_flax_network('u8', feat_domain, act_domain, load, path)
        self.add_flax_network('u9', feat_domain, act_domain, load, path)

        self.add_flax_network('u12', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u13', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u23', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u34', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u27', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u45', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u48', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u58', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u57', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u56', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u67', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u69', feat_bndr, act_bndr, load, path)
        self.add_flax_network('u89', feat_bndr, act_bndr, load, path)


        self.interface12 = interface_function2d(1,1.0,-1.0,self.neural_networks['u12'])
        self.interface21 = interface_function2d(1,1.0,-1.0,self.neural_networks['u12'])

        self.interface13 = interface_function2d(0,1.0,-1.0,self.neural_networks['u13'])
        self.interface31 = interface_function2d(0,1.0,-1.0,self.neural_networks['u13'])

        self.interface23 = interface_function2d(0,1.0,-1.0,self.neural_networks['u23'])
        self.interface32 = interface_function2d(1,1.0,-1.0,self.neural_networks['u23'])
        
        self.interface27 = interface_function2d(1,-1.0,1.0,self.neural_networks['u27'])
        self.interface72 = interface_function2d(0,1.0,-1.0,self.neural_networks['u27'])
        
        self.interface34 = interface_function2d(0,-1.0,1.0,self.neural_networks['u34'])
        self.interface43 = interface_function2d(1,1.0,-1.0,self.neural_networks['u34'])
        
        self.interface45 = interface_function2d(0,1.0,-1.0,self.neural_networks['u45'])
        self.interface54 = interface_function2d(0,1.0,-1.0,self.neural_networks['u45'])
        
        self.interface48 = interface_function2d_inv(1,-1.0,1.0,self.neural_networks['u48']) 
        self.interface84 = interface_function2d(0,1.0,-1.0,self.neural_networks['u48'])
        
        self.interface56 = interface_function2d(0,-1.0,1.0,self.neural_networks['u56'])
        self.interface65 = interface_function2d(0,-1.0,1.0,self.neural_networks['u56'])

        self.interface57 = interface_function2d(1,1.0,-1.0,self.neural_networks['u57'])
        self.interface75 = interface_function2d(1,1.0,-1.0,self.neural_networks['u57'])

        self.interface58 = interface_function2d(1,-1.0,1.0,self.neural_networks['u58'])
        self.interface85 = interface_function2d(1,-1.0,1.0,self.neural_networks['u58'])

        self.interface67 = interface_function2d_inv(1,1.0,-1.0,self.neural_networks['u67'])
        self.interface76 = interface_function2d(0,-1.0,1.0,self.neural_networks['u67'])
        
        self.interface69 = interface_function2d_inv(1,-1.0,1.0,self.neural_networks['u69'])
        self.interface96 = interface_function2d(1,1.0,-1.0,self.neural_networks['u69'])
        
        self.interface89 = interface_function2d(0,-1.0,1.0,self.neural_networks['u89'])
        self.interface98 = interface_function2d(0,1.0,-1.0,self.neural_networks['u89'])

        self.add_trainable_parameter('u123',(1,), load_p, path) 
        self.add_trainable_parameter('u458',(1,), load_p, path) 
        self.add_trainable_parameter('u567',(1,), load_p, path) 
        self.add_trainable_parameter('u5689',(1,), load_p, path) 
        self.add_trainable_parameter('u23457',(1,), load_p, path) 
        
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
        
        points['ys4'] = ys
        points['ws4'] = Weights
        points['omega4'], points['G4'], points['K4'] = geoms[3].GetMetricTensors(ys)

        points['ys5'] = ys
        points['ws5'] = Weights
        points['omega5'], points['G5'], points['K5'] = geoms[4].GetMetricTensors(ys)

        points['ys6'] = ys
        points['ws6'] = Weights
        points['omega6'], points['G6'], points['K6'] = geoms[5].GetMetricTensors(ys)

        points['ys7'] = ys
        points['ws7'] = Weights
        points['omega7'], points['G7'], points['K7'] = geoms[6].GetMetricTensors(ys)

        points['ys8'] = ys
        points['ws8'] = Weights
        points['omega8'], points['G8'], points['K8'] = geoms[7].GetMetricTensors(ys)

        points['ys9'] = ys
        points['ws9'] = Weights
        points['omega9'], points['G9'], points['K9'] = geoms[8].GetMetricTensors(ys)
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
        #                                    --------x23457    
        #                                       7
        #------------------------------------------------------------------------------#
        
        alpha = 2 
        
        u = self.neural_networks['u2'].apply(ws['u2'],x) 

        v = ((1 - x[...,1]) * (x[...,1] + 1) * (1 - x[...,0]))[...,None]
        
        w21 = self.interface21(ws['u12'], x) * (1 - x[...,0])[...,None]         
        w23 = self.interface23(ws['u23'], x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]         
        w27 = self.interface27(ws['u27'], x) * (1 - x[...,0])[...,None]         

        w123   = ws['u123']  *( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha
        w23457  = ws['u23457'] *( (x[...,0] + 1) * (1 - x[...,1]) )[...,None]**alpha
        
        w = w21 + w23 + w27
        w = w + w123 + w23457
        output = u*v + w
        return output 
   


    def solution3(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       2 
        #                              23457x--------x123  
        #                                   |        |
        # 3. Domain : Iron Pole           4 |   3    | 1
        #                                   |        |
        #                                    --------    
        #                                       D
        #------------------------------------------------------------------------------#
        alpha = 2 
        
        u = self.neural_networks['u3'].apply(ws['u3'],x) 

        v = ((1 - x[...,0]) * (x[...,0] + 1) * (1 - x[...,1]) * (x[...,1] + 1))[...,None]
        
        w31 = self.interface31(ws['u13'], x) * ((1 - x[...,1]) *  (x[...,1] + 1))[...,None]
        w32 = self.interface32(ws['u23'], x) * ((1 - x[...,0]) *  (x[...,0] + 1))[...,None]
        w34 = self.interface34(ws['u34'], x) * ((1 - x[...,1]) *  (x[...,1] + 1))[...,None]
        
        w123   = ws['u123']  *( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha
        w23457  = ws['u23457'] *( (1 - x[...,0]) * (x[...,1] + 1) )[...,None]**alpha
        
        w = w31 + w32 + w34
        w = w + w123 + w23457

        output = u*v + w
        return output 
   
    def solution4(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       3 
        #                                    --------x23457  
        #                                   |        |
        # 4. Domain : Iron Yoke           D |   4    | 5
        #                                   |        |
        #                                    --------x458   
        #                                       8
        #------------------------------------------------------------------------------#
        alpha = 2 
        
        u = self.neural_networks['u4'].apply(ws['u4'],x) 

        v = ((1 - x[...,1]) * (x[...,1] + 1) * (1 - x[...,0]) * (x[...,0] + 1))[...,None]
        
        w43 = self.interface43(ws['u34'], x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]
        w45 = self.interface45(ws['u45'], x) * ((1 - x[...,1]) * (x[...,1] + 1)) [...,None]
        w48 = self.interface48(ws['u48'], x) * ((1 - x[...,0]) * (x[...,0] + 1)) [...,None]
        
        w23457  = ws['u23457'] *( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha
        w458  = ws['u458'] *( (x[...,0] + 1) * (1 - x[...,1]) )[...,None]**alpha
        
        w =  w48  + w43 + w45 
        w = w + w23457 + w458

        output = u*v + w
        return output 
    
    def solution5(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       7 
        #                                567x--------x23457  
        #                                   |        |
        # 5. Domain : Air 3               6 |   5    | 4
        #                                   |        |
        #                              5689x --------x458    
        #                                       8 
        #------------------------------------------------------------------------------#
        alpha = 2 
        
        u = self.neural_networks['u5'].apply(ws['u5'],x) 

        v = ((1 - x[...,1]) * (x[...,1] + 1) * (1 - x[...,0]) * (x[...,0] + 1))[...,None]
        
        w57 = self.interface57(ws['u57'], x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]
        w54 = self.interface54(ws['u45'], x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]
        w56 = self.interface56(ws['u56'], x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]        
        w58 = self.interface58(ws['u58'], x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]        

        w23457  = ws['u23457'] * ( ( x[...,0] + 1 ) * ( x[...,1] + 1 ) )[...,None]**alpha
        w5689   = ws['u5689']  * ( ( 1 - x[...,0] ) * ( 1 - x[...,1] ) )[...,None]**alpha
        w567    = ws['u567']   * ( ( 1 - x[...,0] ) * ( x[...,1] + 1 ) )[...,None]**alpha
        w458    = ws['u458']   * ( ( x[...,0] + 1 ) * ( 1 - x[...,1] ) )[...,None]**alpha
        
        w = w56 + w54 + w57 + w58
        w = w + w23457 + w567 + w458 + w5689

        output = u*v + w
        return output 
    
    def solution6(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       7 
        #                                567x --------  
        #                                   |        |
        # 6. Domain : Air 3               5 |   6    | N
        #                                   |        |
        #                               5689x--------    
        #                                       9
        #------------------------------------------------------------------------------#
        alpha = 2 
        
        u = self.neural_networks['u6'].apply(ws['u6'],x) 

        v = ((x[...,1] + 1) * (1 - x[...,1]) * (x[...,0] + 1))[...,None]
        
        w65 = self.interface65(ws['u56'], x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]
        w67 = self.interface67(ws['u67'], x) *  (x[...,0] + 1) [...,None]
        w69 = self.interface69(ws['u69'], x) *  (x[...,0] + 1) [...,None]
        
        w567  = ws['u567'] *( (1 - x[...,0]) * (x[...,1] + 1) )[...,None]**alpha
        w5689   = ws['u5689']  * ( ( 1 - x[...,0] ) * ( 1 - x[...,1] ) )[...,None]**alpha

        w = w65 + w67 + w69
        w = w + w567 + w5689

        output = u*v + w
        return output 


    def solution7(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       5 
        #                                567x--------x23457
        #                                   |        |
        # 7. Domain : Air 4               6 |   7    | 2
        #                                   |        |
        #                                    --------    
        #                                       N
        #------------------------------------------------------------------------------#
        alpha = 2 
        
        u = self.neural_networks['u7'].apply(ws['u7'],x) 

        v = ((x[...,0] + 1) * (1 - x[...,0]) * (1 - x[...,1]))[...,None]
        
        w72 = self.interface72(ws['u27'], x) * ((1 - x[...,1]))[...,None]
        w75 = self.interface75(ws['u57'], x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]
        w76 = self.interface76(ws['u67'], x) *  (1 - x[...,1]) [...,None]
        
        w23457  = ws['u23457'] *( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha
        w567   = ws['u567']    *( (1 - x[...,0]) * (x[...,1] + 1) )[...,None]**alpha
        
        w = w72 + w75 + w76 
        w = w + w23457 + w567

        output = u*v + w
        return output 

    def solution8(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       D 
        #                                    --------
        #                                   |        |
        # 7. Domain : Air 4               9 |   8    | 4
        #                                   |        |
        #                               5689x--------x458    
        #                                       5
        #------------------------------------------------------------------------------#
        alpha = 2 
        
        u = self.neural_networks['u8'].apply(ws['u8'],x) 

        v = ((1 - x[...,1]) * (x[...,1] + 1) * (1 - x[...,0]) * (x[...,0] + 1))[...,None]
        
        w84 = self.interface84(ws['u48'], x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]        
        w85 = self.interface85(ws['u58'], x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]
        w89 = self.interface89(ws['u89'], x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]

        
        w5689   = ws['u5689']  * ( ( 1 - x[...,0] ) * ( 1 - x[...,1] ) )[...,None]**alpha
        w458    = ws['u458']   * ( ( x[...,0] + 1 ) * ( 1 - x[...,1] ) )[...,None]**alpha

        w = w84 + w85 + w89 
        w = w + w5689 + w458

        output = u*v + w
        return output 
    
    def solution9(self, ws, x):
        #------------------------------------------------------------------------------#
        #                                       6 
        #                                    --------x5689
        #                                   |        |
        # 7. Domain : Air 4               N |   9    | 8
        #                                   |        |
        #                                    --------    
        #                                       D
        #------------------------------------------------------------------------------#
        alpha = 2 
        
        u = self.neural_networks['u9'].apply(ws['u9'],x) 

        v = ( (1 - x[...,1]) * (x[...,1] + 1) * (1 - x[...,0]) )[...,None]
        
        w98 = self.interface98(ws['u89'], x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]
        w96 = self.interface96(ws['u69'], x) * (1 - x[...,0])[...,None]
        
        w5689  = ws['u5689'] * ( ( x[...,0] + 1 ) * ( x[...,1] + 1 ) )[...,None]**alpha
        
        w = w98 + w96
        w = w + w5689

        output = u*v + w
        return output 




    def loss_pde(self, ws, points):
        grad1 = src.operators.gradient(lambda x : self.solution1(ws,x))(points['ys1'])[...,0,:]
        grad2 = src.operators.gradient(lambda x : self.solution2(ws,x))(points['ys2'])[...,0,:]
        grad3 = src.operators.gradient(lambda x : self.solution3(ws,x))(points['ys3'])[...,0,:]
        grad4 = src.operators.gradient(lambda x : self.solution4(ws,x))(points['ys4'])[...,0,:]
        grad5 = src.operators.gradient(lambda x : self.solution5(ws,x))(points['ys5'])[...,0,:]
        grad6 = src.operators.gradient(lambda x : self.solution6(ws,x))(points['ys6'])[...,0,:]
        grad7 = src.operators.gradient(lambda x : self.solution7(ws,x))(points['ys7'])[...,0,:]
        grad8 = src.operators.gradient(lambda x : self.solution8(ws,x))(points['ys8'])[...,0,:]
        grad9 = src.operators.gradient(lambda x : self.solution9(ws,x))(points['ys9'])[...,0,:]
        
        lpde1 = 0.5*1/(self.mu0)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad1,points['K1'],grad1), points['ws1'])

        lpde2 = 0.5*1/(self.mu0)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad2,points['K2'],grad2), points['ws2'])

        lpde6 = 0.5*1/(self.mu0)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad6,points['K6'],grad6), points['ws6'])

        lpde7 = 0.5*1/(self.mu0)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad7,points['K7'],grad7), points['ws7'])

        lpde3 = 0.5*1/(self.mu0*self.mur)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad3,points['K3'],grad3), points['ws3']) 

        lpde4 = 0.5*1/(self.mu0*self.mur)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad4,points['K4'],grad4), points['ws4']) 

        lpde8 = 0.5*1/(self.mu0*self.mur)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad8,points['K8'],grad8), points['ws8']) 

        lpde9 = 0.5*1/(self.mu0*self.mur)*jnp.dot(jnp.einsum('mi,mij,mj->m',grad9,points['K9'],grad9), points['ws9']) 

        lpde5 = 0.5*1/self.mu0*jnp.dot(jnp.einsum('mi,mij,mj->m',grad5,points['K5'],grad5), points['ws5'])  - jnp.dot(self.J0*self.solution5(ws,points['ys5']).flatten()*points['omega5'], points['ws5'])
        return lpde1 + lpde2 + lpde3 + lpde4 + lpde5 + lpde6 + lpde7 + lpde8 + lpde9


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
n_epochs = 500                                                             # Number of optimization epochs
path_coor = './fem_ref/coordinates.csv'                                     # Path to coordinates to evaluate the NN solution
path_refs = './parameters/quad/mu_2k/ref_values.csv'                        # FEM reference solution
# meshfile = './fem_ref/fenicsx_mesh/quad_simple/quad_simple' 

opt_init, opt_update, get_params = optimizers.adamax(step_size=stepsize)    # Instantiate the optimizer
opt_state = opt_init(weights)                                               # Initialize the optimizer with the NN weights
params = get_params(opt_state)                                              # Retrieve the trainable weights for the optimizer as a dict
loss_grad = jax.jit(lambda ws, pts: (model.loss(ws, pts), jax.grad(model.loss)(ws, pts))) # JIT compile the loss function before training

bnd_samples = sample_bnd(1000)
key = jax.random.PRNGKey(1223435)
points = model.get_points_MC(batch_size, key)                               # Generate the MC samples
output_4 = model.solution4(params, bnd_samples[1])
output_8 = model.solution8(params, bnd_samples[0])
plt.figure()
plt.plot(output_4, label = 'u48')
plt.plot(np.flip(output_8), label = 'u84')
plt.legend()
plt.savefig('./bnd_48.png')

key = jax.random.PRNGKey(1223435)
points = model.get_points_MC(batch_size, key)                               # Generate the MC samples

evaluate_error(model, params, evaluate_models, [0,1,2,3,4,5,6,7,8], geoms, meshfile)
exit()

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

evaluate_error(model, params, evaluate_models, [0,1,2,3,4,5,6,7,8], geoms, meshfile)
