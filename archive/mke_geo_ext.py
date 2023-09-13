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
rnd_key = jax.random.PRNGKey(1234)

#%% Geometry parametrizations

def create_geometry(key, scale = 1):
    scale = scale
    Do = 142e-3                                                            
    Di = 51e-3                                                            
    #Di = 101e-3                                                            
    #hi = 20e-3                                                             
    hi = 40e-3                                                             
    #bli = 3e-3                                                             
    bli = 6e-3                                                             
    #Dc = 3.27640e-2                                                          
    Dc = 5e-2                                                          
    hc = 7.55176e-3                                                           
    ri = 20e-3                                                           
    ri = 10e-3                                                           
    blc = hi-hc                                                           
    rm = (Dc*Dc+hc*hc-ri*ri)/(Dc*np.sqrt(2)+hc*np.sqrt(2)-2*ri)
    R = rm-ri
    O = np.array([rm/np.sqrt(2),rm/np.sqrt(2)])
    alpha1 = -np.pi*3/4       
    alpha2 = np.math.asin((hc-rm/np.sqrt(2))/R)
    alpha = np.abs(alpha2-alpha1)
    
    A = np.array([[O[0] - ri/np.sqrt(2), O[1] - ri/np.sqrt(2)], [O[0] - Dc, O[1] - hc]])
    b = np.array([[A[0,0]*ri/np.sqrt(2)+A[0,1]*ri/np.sqrt(2)],[A[1,0]*Dc+A[1,1]*hc]])
    C = np.linalg.solve(A,b)

    angle = -np.pi/8
    rotation_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    print(rotation_mat.shape)


    knots1 = np.array([[0.142, 0.05881833],               \
                       [0.10040916, 0.10040916],          \
                       [0.07002297, 0.07002297],          \
                       [0.05002297, 0.05002297],          \
                       [0.00707107, 0.00707107] ])
    
    p1 = 0.10040916
    p2 = 0.07002297
    p3 = 0.05002297
    p4 = 0.00707107


    p1 = 0.12
    p2 = 0.1
    p3 = 0.07
    p4 = 0.01

    knots_outer = np.array([[p1, p2, p3, p4]]).T
    knots1 = np.concatenate((knots_outer, knots_outer), axis = 1) # f(x)=x for the y coordinate
    knot_bnd = np.matmul(rotation_mat, knots1[0,:])
    knots1 = np.concatenate((knot_bnd[np.newaxis,:], knots1)) 

    #knots1 = np.array([[Do,Do * np.tan(np.pi/8)],[Do/np.sqrt(2),Do/np.sqrt(2)], \
    #                 [rm+0.03,rm+0.03], \
    #                 [(rm+ri),(rm+ri)], \
    #                 [ri/np.sqrt(2),ri/np.sqrt(2)]])
    #knots2 = np.array([[Di,hi-bli],[Di-bli,hi], \
    #                   [Dc+blc,hi],[(1/2)*(2*Dc+blc)+0.01,(1/2)*(hc+hi)+0.01],[Dc,hc]])
    
    knots2 =  np.array([[0.101, 0.034],               \
                        [0.095, 0.04],                \
                        [0.08244824, 0.04],           \
                        [0.07622412, 0.03377588],     \
                        [0.05, 0.00755176] ] )
    h = 0.03
    d4x = p3 + h/np.sqrt(2)
    d4y = p3 - h/np.sqrt(2)
    delx = 0.005
    dely = delx/2
    knots2 = np.array([[d4x + 5*delx, d4y-dely],[d4x + 4*delx, d4y + 2*dely],[d4x + delx, d4y + 2*dely] ,[d4x, d4y], [0.05, 0.00755176]])

    knots3 = (knots1+knots2)/2

    knots3[-1,:] = C.flatten()
    knots = np.concatenate((knots1[None,...],knots3[None,...],knots2[None,...]),0)

    plot_knots = np.reshape(knots, (knots.shape[0]*knots.shape[1],2))
    plt.scatter(plot_knots[:,0], plot_knots[:,1], c= 'r')
    
    weights = np.ones(knots.shape[:2])
    weights[1,-1] = np.sin((np.pi-alpha)/2)
    basis1 = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
    basis2 = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0,0.33,1]),1)

    geom1 = src.geometry.PatchNURBSParam([basis1, basis2], knots, weights, 0, 2, key)

    return  geom1

#%% Instantiate geometry parametrizations

geom1 = create_geometry(rnd_key)
pts,_ = geom1.importance_sampling(1000)
plt.scatter(pts[:,0], pts[:,1], s = 1)

ys = jax.random.uniform(rnd_key, (1000, 2))
ys = 2*ys - 1
ys = ys.at[:,0].set(1)
#ys = ys.at[:,1].set(1)
bnd = geom1.__call__(ys)
plt.scatter(bnd[:,0], bnd[:,1], c='r')
plt.show()
plt.show()