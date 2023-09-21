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
from post_processing import evaluate_error



# Geometry parametrizations
iron_pole, iron_yoke, iron_yoke_r_mid, iron_yoke_r_low, air_1, air_2, air_3, current  = create_geometry(rnd_key)

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
        nl_bndr = 8 
        load =True 
        load_p = True 

        feat_domain = [2, nl, nl, nl, 1] 
        feat_bndr = [1, nl_bndr, nl_bndr, nl_bndr, 1] 

        # 1 PoleTip
        self.add_flax_network('u1', feat_domain, load)
        # 2 IronYoke
        self.add_flax_network('u2', feat_domain, load)
        # 3 IronYoke Right Middle
        self.add_flax_network('u3', feat_domain, load)
        # 4 IronYoke Right Lower
        self.add_flax_network('u4', feat_domain, load)
        # 5 Air1
        self.add_flax_network('u5', feat_domain, load)
        # 6 Air2 
        self.add_flax_network('u6', feat_domain, load)
        # 7 Air3
        self.add_flax_network('u7', feat_domain, load)
        # 8 Current
        self.add_flax_network('u8', feat_domain, load)

        # Interfaces to PoleTip 
        self.add_flax_network('u15', feat_bndr, load)
        self.add_flax_network('u16', feat_bndr, load)
        self.add_flax_network('u12', feat_bndr, load)

        # Interfaces to Iron Yoke
        self.add_flax_network('u28', feat_bndr, load)
        self.add_flax_network('u23', feat_bndr, load)

        # Interfaces to Iron Yoke Right Middle
        self.add_flax_network('u38', feat_bndr, load)
        self.add_flax_network('u34', feat_bndr, load)

        # Interfaces to Iron Yoke Right Lower
        self.add_flax_network('u47', feat_bndr, load)

        # Interfaces to Air1
        self.add_flax_network('u56', feat_bndr, load)
        
        # Interfaces to Air2
        self.add_flax_network('u67', feat_bndr, load)
        self.add_flax_network('u68', feat_bndr, load)
        
        # Interfaces to Air3
        self.add_flax_network('u78', feat_bndr, load)
        
        self.add_flax_network('u2_0.3', feat_bndr, load)
        self.add_flax_network('u2_0.7', feat_bndr, load)

        self.add_flax_network('u8_0.3', feat_bndr, load)
        self.add_flax_network('u8_0.7', feat_bndr, load)

        # Jump Functions in IronYoke and Current domain
        #self.add_neural_network('u2_0.3',stax.serial(block_first, block, block, stax.Dense(1)),(-1,1))
        #self.add_neural_network('u2_0.7',stax.serial(block_first, block, block, stax.Dense(1)),(-1,1))
        
        # self.add_neural_network('u8_0.3',stax.serial(block_first, block, block, stax.Dense(1)),(-1,1))
        # self.add_neural_network('u8_0.7',stax.serial(block_first, block, block, stax.Dense(1)),(-1,1))

        self.add_trainable_parameter('u156',(1,), load_p) 
        self.add_trainable_parameter('u238',(1,), load_p) 
        self.add_trainable_parameter('u567',(1,), load_p) 
        self.add_trainable_parameter('u678',(1,), load_p) 
        self.add_trainable_parameter('u1268',(1,), load_p) 
        self.add_trainable_parameter('u3478',(1,), load_p) 

        self.add_trainable_parameter('u28_p0.33',(1,), load_p)
        self.add_trainable_parameter('u28_n0.33',(1,), load_p)

        self.add_trainable_parameter('u87_p0.33',(1,), load_p)
        self.add_trainable_parameter('u87_n0.33',(1,), load_p)
        

        # Domains: 1: PoleTip, 2: IronYoke, 3: IronYoke Right Middle, 4. IronYoke Right Lower, 
        #          5. Air1,    6. Air2,     7. Air3,                  8. Current
        #------------------------------------------------------------------------------#
        # PoleTip -> IronYoke       |   NN_{12}(x)*-1/2(y-1)  => (1, -1, 1)   
        self.interface12 = interface_function2d(1, -1.0, 1.0,self.neural_networks['u12'])
        # IronYoke -> PoleTip       |   NN_{12}(x)*1/2(y+1)   => (1, 1, -1) 
        self.interface21 = interface_function2d(1, 1.0, -1.0,self.neural_networks['u12'])

        self.interface15 = interface_function2d(1, 1.0, -1.0,self.neural_networks['u15'])
        # PoleTip -> Air1          |   NN_{15}(x)*1/2(y+1)   => (1, 1, -1) 
        self.interface51 = interface_function2d(0, 1.0, -1.0,self.neural_networks['u15'])
        # ### Air1 -> PoleTip      |   NN_{15}(y)*1/2(x+1)   => (0, 1, -1) 

        self.interface16 = interface_function2d(0, 1.0, -1.0,self.neural_networks['u16'])
        # PoleTip -> Air2      |   NN_{16}(y)*1/2(x+1)   => (0, 1, -1) 
        self.interface61 = interface_function2d(0, -1.0, 1.0,self.neural_networks['u16'])
        # Air2 -> PoleTip      |   NN_{16}(y)*-1/2(x-1)  => (0, -1, 1) 

        #------------------------------------------------------------------------------#
        # IronYoke -> IronYoke Right Middle   |   NN(x)* -1/2(y-1)  => (1, -1, 1)
        self.interface23 = interface_function2d(1, -1.0, 1.0,self.neural_networks['u23'])
        # IronYoke Right Middle -> IronYoke   |   NN(y)* -1/2(x-1) => (0, -1, 1)
        self.interface32 = interface_function2d(0, -1.0, 1.0,self.neural_networks['u23'])

        # IronYoke -> Current   |   NN(y)* 1/2(x+1)  => (0, 1, -1) 
        self.interface28 = interface_function2d(0, 1.0, -1.0,self.neural_networks['u28'])
        # Current -> IronYoke   |   NN(y)* -1/2(x-1) => (0, -1, 1) 
        self.interface82 = interface_function2d(0, -1.0, 1.0,self.neural_networks['u28'])
        
        #------------------------------------------------------------------------------#
        # IronYoke Right Middle -> Current   |   NN(x)*1/2(y+1) => (1, 1, -1) 
        self.interface38 = interface_function2d(1, 1.0, -1.0,self.neural_networks['u38'])
        # Current -> IronYoke Right Middle   |   NN(x)* -1/2(y-1) => (1, -1, 1) 
        self.interface83 = interface_function2d(1, -1.0, 1.0,self.neural_networks['u38'])
        
        # IronYoke Right Middle -> IronYoke Right Lower |   NN_{34}(y)*1/2(x+1) => (0, 1, -1) 
        self.interface34 = interface_function2d(0, 1.0, -1.0,self.neural_networks['u34'])
        # IronYoke Right Lower -> IronYoke Right Middle |   NN_{34}(y)* -1/2(x-1) => (0, -1, 1) 
        self.interface43 = interface_function2d(0, -1.0, 1.0,self.neural_networks['u34'])

        #------------------------------------------------------------------------------#
        # IronYoke Right Lower -> Air3   |   NN(x)*1/2(y+1)   => (1, 1, -1) 
        self.interface47 = interface_function2d(1, 1.0, -1.0,self.neural_networks['u47'])
        # Air3 -> IronYoke Right Lower   |   NN(x)* -1/2(y-1) => (1, -1, 1) 
        self.interface74 = interface_function2d(1, -1.0, 1.0,self.neural_networks['u47'])
        
        #------------------------------------------------------------------------------#
        # Air1 -> Air2   |   NN(x)* 1/2(y+1)   => (1, 1, -1) 
        self.interface56 = interface_function2d(1, 1.0, -1.0,self.neural_networks['u56'])
        # Air2 -> Air1   |   NN(x)* 1/2(y+1)  => (1, 1, -1) 
        self.interface65 = interface_function2d_inv(1, 1.0, -1.0,self.neural_networks['u56'])
        
        #------------------------------------------------------------------------------#
        # Air2 -> Air3   |   NN(y)*1/2(x+1)   => (0, 1, -1) 
        self.interface67 = interface_function2d(0, 1.0, -1.0,self.neural_networks['u67'])
        # Air3 -> Air2   |   NN(x)* 1/2(y+1)  => (1, 1, -1) 
        self.interface76 = interface_function2d(1, 1.0, -1.0,self.neural_networks['u67'])
        
        # Air2 -> Current   |   NN(x)*-1/2(y-1)   => (1, -1, 1) 
        self.interface68 = interface_function2d(1, -1.0, 1.0,self.neural_networks['u68'])
        # Current -> Air2   |   NN(x)* 1/2(y+1)   => (1, 1, -1) 
        self.interface86 = interface_function2d(1, 1.0, -1.0,self.neural_networks['u68'])
        
        #------------------------------------------------------------------------------#
        # Air3 -> Current   |   NN(y)*-1/2(x-1)   => (0, -1, 1) 
        self.interface78 = interface_function2d(0, -1.0, 1.0,self.neural_networks['u78'])
        # Current -> Air3   |   NN(y)* 1/2(x+1)   => (0, 1, -1) 
        self.interface87 = interface_function2d(0, 1.0, -1.0,self.neural_networks['u78'])
        
        
        # Functions defining the compactly supported solution
        self.jump1 = jump_function2d(0, -0.33, self.neural_networks['u2_0.3'])
        self.jump2 = jump_function2d(0,  0.33, self.neural_networks['u2_0.7'])

        self.jump3 = jump_function2d(0, -0.33, self.neural_networks['u8_0.3'])
        self.jump4 = jump_function2d(0,  0.33, self.neural_networks['u8_0.7'])

        #self.mu0 = 0.001
        self.mu0 = 1
        self.mur = 1
        # self.mur =1 
        self.J0 =  1000

        self.k1 = 0.001
        self.k2 = 1.65/5000
        self.k3 = 0.5
        # 10 000
        # self.points = self.get_points_MC(10000, self.key)
        

    def get_points_MC(self, N, key):        
        points = {}

        ys = jax.random.uniform(key ,(N,2))*2-1
        Weights = jnp.ones((N,))*4/ys.shape[0]


        points['ys1'] = ys
        points['ws1'] = Weights
        points['omega1'], points['G1'], points['K1'] = iron_pole.GetMetricTensors(ys)
       
        points['ys2'] = ys
        points['ws2'] = Weights
        points['omega2'], points['G2'], points['K2'] = iron_yoke.GetMetricTensors(ys)
        
        points['ys3'] = ys
        points['ws3'] = Weights
        points['omega3'], points['G3'], points['K3'] = iron_yoke_r_mid.GetMetricTensors(ys)
       
        points['ys4'] = ys
        points['ws4'] = Weights
        points['omega4'], points['G4'], points['K4'] = iron_yoke_r_low.GetMetricTensors(ys)

        points['ys5'] = ys
        points['ws5'] = Weights
        points['omega5'], points['G5'], points['K5'] = air_1.GetMetricTensors(ys)
       
        points['ys6'] = ys
        points['ws6'] = Weights
        points['omega6'], points['G6'], points['K6'] = air_2.GetMetricTensors(ys)
        
        points['ys7'] = ys
        points['ws7'] = Weights
        points['omega7'], points['G7'], points['K7'] = air_3.GetMetricTensors(ys)
       
        points['ys8'] = ys
        points['ws8'] = Weights
        points['omega8'], points['G8'], points['K8'] = current.GetMetricTensors(ys)
        return points



    def solution1(self, ws, x):
        # 1. Domain : PoleTip 
        alpha = 2

        # Solution on the domain
        u = self.neural_networks['u1'].apply(ws['u1'],x) 
        
        # Ansatz function: v(x,y) = (1-x)*(x+1)*(1+y)*(1-y)
        v = ((1 - x[...,0]) * (x[...,0] + 1) * (1 - x[...,1]) * (x[...,1] + 1))[...,None]

        # Interfaces along other domains
        w12 = self.interface12(ws['u12'],x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]
        w15 = self.interface15(ws['u15'],x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]
        w16 = self.interface16(ws['u16'],x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]

        # Interfaces along other domains
        w156  = ws['u156'] *( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha 
        w1268 = ws['u1268']*( (x[...,0] + 1) * (1 - x[...,1]) )[...,None]**alpha 
        
        w = w12 + w15 + w16 + w156 + w1268   
        return u * v + w

    def solution2(self, ws, x):
        # 2. Domain : IronYoke
        alpha = 2
        
        # Solution on the domain
        u = self.neural_networks['u2'].apply(ws['u2'],x) + self.jump1(ws['u2_0.3'], x) + self.jump2(ws['u2_0.7'], x)
        
        # Ansatz function which vanishes on the boundary (pyramid)  
        v = ((1 - x[...,0])*(x[...,0] + 1) * (1 - x[...,1])*(x[...,1] + 1))[...,None]
        
        # Function: v(x,y) = (1-x)*(1+x)*(1-y)*(1+y)

        # Interface functions with Ansatz functions depending on location
        # ws['u12'] -> IronYoke - PoleTip                  |      (1-x)*(1+x)
        # ws['u23'] -> IronYoke - IronYoke Right Middle    |      (1-x)*(1+x)
        # ws['u28'] -> IronYoke - Current                  |      (1-y)*(1+y)
 
        w21 = self.interface21(ws['u12'],x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]
        w23 = self.interface23(ws['u23'],x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]
        w28 = (self.interface28(ws['u28'],x)                        \
               + ExpHat(x[...,1] + 0.33)[...,None]*ws['u28_n0.33']  \
               + ExpHat(x[...,1] - 0.33)[...,None]*ws['u28_p0.33']  \
                )*(1 - x[...,1])[...,None] * (x[...,1] + 1)[...,None]
        
        # Interface functions for coil domain
        #------------------------------------------------------------------------------#
        # w21 = NN_{12}(x)*1/2(y+1) * (x+1)(1-x)               |   
        # w28 = (                                              |   
        #           NN_{28}(y)* 1/2(x+1)                       |
        #         + exp(|y-0.33|) * u_{28l}                    |   
        #         + exp(|y+0.33|) * u_{28r}                    |
        #                                       )*(1-y)(y+1)   |   
        # w23 = NN_{23}(x)*-1/2(y-1) * (1-x)(1+x)              |   
        
        w1268 = ws['u1268'] * ( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha
        w238  =  ws['u238'] * ( (x[...,0] + 1) * (1 - x[...,1]) )[...,None]**alpha
        
        # Function coinciding on the three subdomains
        #------------------------------------------------------------------------------#
        # w1268 = u_{1268} * ((x+1)*(y+1))^2   
        # w156  = u_{156} * ((x+1)*(1-y))^2   

        w = w28 + w21 + w23 + w1268 + w238
        return u * v + w
    
    def solution3(self, ws, x):
        # 3. Domain : IronYoke Right Middle
        alpha = 2

        u = self.neural_networks['u3'].apply(ws['u3'],x)
        # Ansatz Function: v(x,y) = (1-x)*(x+1)*(1-y)*(y+1)
        #------------------------------------------------------------------------------#
        v = ((1 - x[...,0]) * (x[...,0] + 1) * (1 - x[...,1]) * (x[...,1] + 1))[...,None]
        
        # Interface functions for IronYoke Right Middle 
        #------------------------------------------------------------------------------#
        w34 = self.interface34(ws['u34'],x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]
        w38 = self.interface38(ws['u38'],x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]
        w32 = self.interface32(ws['u23'],x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]
        #------------------------------------------------------------------------------#
        # w34 = NN_{34}(y) *  1/2(x+1) * (1-y) * (y+1)     |   
        # w38 = NN_{38}(y) *  1/2(y+1) * (1-x) * (x+1)     |
        # w32 = NN_{23}(y) * -1/2(x-1) * (1-y) * (y+1)     |


        # Function coinciding on multiple subdomains
        #------------------------------------------------------------------------------#
        w238  = ws['u238']  *  ( (1 - x[...,0]) * (x[...,1] + 1) )[...,None]**alpha
        w3478 = ws['u3478'] *  ( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha
        #------------------------------------------------------------------------------#
        # w238  = u_{238}  * ((1-x)*(y+1))^2    |
        # w3478 = u_{3478} * ((x+1)*(y+1))^2   |

        w = w32 + w34 + w38 + w238 + w3478
        return u * v + w
    
    def solution4(self, ws, x):
        # 4. Domain : IronYoke Right Lower
        alpha = 2

        u = self.neural_networks['u4'].apply(ws['u4'],x)
        # Ansatz Function: v(x,y) = (x+1)*(1-y)*(y+1)
        #------------------------------------------------------------------------------#
        v = ( (x[...,0] + 1) * (1 - x[...,1]) * (x[...,1] + 1) )[...,None]
        
        # Interface functions for IronYoke Right Lower 
        #------------------------------------------------------------------------------#
        w47 = self.interface47(ws['u47'],x) * ( (x[...,0] + 1))[...,None]
        w43 = self.interface43(ws['u34'],x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]
        #------------------------------------------------------------------------------#
        # w47 = NN_{47}(x) *  1/2(y+1) * (x+1)             |   
        # w43 = NN_{34}(y) * -1/2(x-1) * (1-y) * (y+1)     |


        # Function coinciding on multiple subdomains
        #------------------------------------------------------------------------------#
        w3478 = ws['u3478'] *  ( (1 - x[...,0]) * (x[...,1] + 1) )[...,None]**alpha
        #------------------------------------------------------------------------------#
        # w3478 = u_{3478} * ((1-x)*(y+1))^2   |

        w = w47 + w43 + w3478
        return u * v + w



    def solution5(self, ws, x):
        # 5. Domain : Air1
        alpha = 2

        u = self.neural_networks['u5'].apply(ws['u5'],x)
        
        # Ansatz Function: v(x,y) = (1-x)*(-)*(1-y)*(1+y)
        v = ((1 - x[...,1]) * (x[...,1] + 1) * (1 - x[...,0]))[...,None]
        
        w56 = self.interface56(ws['u56'],x) * ((x[...,0] + 1) * (1 - x[...,0]) )[...,None]
        # w56 = self.interface56(ws['u56'],x) * ((1 - x[...,0]) )[...,None]


        w51 = self.interface51(ws['u15'],x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]

        # Interface functions for Air1
        #------------------------------------------------------------------------------#
        # w_56 = NN_{56}(x)* 1/2(y+1) * (1-x)                   |   
        # w_51 = NN_{15}(y)* 1/2(x+1) * (1-y)(1+y)              |  

        # Original: w156 = ws['u156']*( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha
        w156 = ws['u156']*( (1 + x[...,0]) * ( 1 + x[...,1]) )[...,None]**alpha
        w567 = ws['u567']*( (1-x[...,0]) * (x[...,1]+1) )[...,None]**alpha
        
        # Function coinciding on the three subdomains
        #------------------------------------------------------------------------------#
        # w156 = u_{156} * ((x+1)*(y+1))^2   |

        w = w51 + w56 + w156 + w567
        return u * v + w
        


    
    def solution6(self, ws, x):
        # 6. Domain : Air2 
        alpha = 2

        u = self.neural_networks['u6'].apply(ws['u6'],x)
        # Ansatz Function: v(x,y) = (1-x)*(x+1)*(1-y)*(y+1)
        #------------------------------------------------------------------------------#
        v = ((1 - x[...,0]) * (x[...,0] + 1) * (1 - x[...,1]) * (x[...,1] + 1))[...,None]
        # v = ((x[...,0] + 1) * (1 - x[...,1]) * (x[...,1] + 1))[...,None]
        
        # Interface functions for IronYoke Right Middle 
        #------------------------------------------------------------------------------#
        w65 =  self.interface65(ws['u56'],x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]
        # w65 = self.interface65(ws['u56'],x) * (x[...,0] + 1)[...,None]

        w67 = self.interface67(ws['u67'],x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]
        # w67 = self.interface67(ws['u67'],x) * ((x[...,1] + 1))[...,None]

        w68 = self.interface68(ws['u68'],x) * ((1 - x[...,0]) * (x[...,0] + 1))[...,None]
        w61 = self.interface61(ws['u16'],x) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]
        #------------------------------------------------------------------------------#
        # w65 = NN_{56}(x) *  1/2(y+1) * (1-x) * (x+1)     |   
        # w67 = NN_{38}(y) *  1/2(x+1) * (1-y) * (y+1)     |
        # w68 = NN_{68}(x) * -1/2(y-1) * (1-x) * (x+1)     |
        # w61 = NN_{16}(y) * -1/2(x-1) * (1-y) * (y+1)     |

        # Function coinciding on multiple subdomains
        #------------------------------------------------------------------------------#
        w567  = ws['u567']   *  ( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha
        w1268 = ws['u1268']  *  ( (1 - x[...,0]) * (1 - x[...,1]) )[...,None]**alpha
        w678  = ws['u678']   *  ( (x[...,0] + 1) * (1 - x[...,1]) )[...,None]**alpha
        # Original w156  = ws['u156']   *  ( (1 - x[...,0]) * (x[...,1] + 1) )[...,None]**alpha
        w156  = ws['u156']   *  ( (1 - x[...,0]) * (x[...,1] + 1) )[...,None]**alpha
        #------------------------------------------------------------------------------#
        # w567  = u_{567}  * ((x+1)*(y+1))^2    |
        # w3467 = u_{3467} * ((1-x)*(1-y))^2    |
        # w678  = u_{678}  * ((1-x)*(y+1))^2    |

        w = w67 + w65 + w68 + w61 + w156 + w1268 + w678 + w567
        return u * v + w


    def solution7(self, ws, x):
        # 7. Domain: Air3
        alpha = 2

        u = self.neural_networks['u7'].apply(ws['u7'],x)
        # Ansatz Function: v(x,y) = (x+1)*(1-y)*(y+1)
        #------------------------------------------------------------------------------#
        v = ((x[...,0] + 1) * (1 - x[...,1]) * (x[...,1] + 1))[...,None]
        
        # Interface functions for IronYoke Right Middle 
        #------------------------------------------------------------------------------#
        # w76 = self.interface76(ws['u67'],x) * ((x[...,0] + 1))[...,None]
        w76 = self.interface76(ws['u67'],x) * ((x[...,0] + 1) * (1 - x[...,0]))[...,None]
        w74 = self.interface74(ws['u47'],x) * ((x[...,0] + 1))[...,None]
        w78 = (self.interface78(ws['u78'],x)                         \
                #+ ExpHat(x[...,1] + 0.33)[...,None] * ws['u87_n0.33']  \
                #+ ExpHat(x[...,1] - 0.33)[...,None] * ws['u87_p0.33']  \
                ) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]
        #------------------------------------------------------------------------------#
        # w76 = NN_{67}(x)  *  1/2(y+1) * (x+1)                      |
        # w74 = NN_{47}(x)  * -1/2(y-1) * (x+1)                      |
        # w78 = (NN_{78}(y) *  (-1/2(x-1) + ...) * (1-y) * (y+1)     |


        # Function coinciding on multiple subdomains
        #------------------------------------------------------------------------------#
        w567  = ws['u567']  *  ( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha
        w3478 = ws['u3478'] *  ( (1 - x[...,0]) * (1 - x[...,1]) )[...,None]**alpha
        w678  = ws['u678']  *  ( (1 - x[...,0]) * (x[...,1] + 1) )[...,None]**alpha

        
        #------------------------------------------------------------------------------#
        # w567  = u_{567}  * ((x+1)*(y+1))^2    |
        # w3478 = u_{3478} * ((1-x)*(1-y))^2    |
        # w678  = u_{678}  * ((1-x)*(y+1))^2    |

        w = w76 + w74 + w78 + w3478 + w678 + w567
        return u * v + w



    def solution8(self, ws, x):
        # 8. Domain : Copper (Coil)
        alpha = 2

        # Inner degrees of freedom:
        u = self.neural_networks['u8'].apply(ws['u8'],x) + self.jump3(ws['u8_0.3'], x) + self.jump4(ws['u8_0.7'], x)
        
        # Ansatz Function for inner dofs
        v = ((1 - x[...,1]) * (x[...,1] + 1) * (1 - x[...,0]) * (x[...,0] + 1))[...,None]
        # v(x,y) = (1-x)*(x+1)*(1-y)*(1+y)

        
        w86 =  self.interface86(ws['u68'],x) * ((x[...,0] + 1) * (1 - x[...,0]))[...,None] 

        w87 = (self.interface87(ws['u78'],x)                           \
                #+ ExpHat(x[...,1] + 0.33)[...,None] * ws['u87_n0.33']  \
                #+ ExpHat(x[...,1] - 0.33)[...,None] * ws['u87_p0.33']  \
                    ) * ((1 - x[...,1]) * (x[...,1] + 1))[...,None]

        w83 = self.interface83(ws['u38'],x) * ((x[...,0] + 1) * (1 - x[...,0]))[...,None]
        w82 = (self.interface82(ws['u28'],x)                         \
                + ExpHat(x[...,1] + 0.33)[...,None]*ws['u28_n0.33']  \
                + ExpHat(x[...,1] - 0.33)[...,None]*ws['u28_p0.33'] \
                ) * ((x[...,1] + 1) * (1 - x[...,1]))[...,None]
        
        
        # Interface functions for Copper (Coil) domain
        #------------------------------------------------------------------------------#
        # w86 = NN_{68}(x) * 1/2(y+1) * (x+1) * (1-x)                    |   
        # w87 = (                                                        |  
        #           NN_{78}(y)* 1/2(x+1)                                 |
        #         + exp(|y - 0.33|) * u_{13l}                            |   
        #         + exp(|y + 0.33|) * u_{13r}                            |
        #                                       )*(1-y)(y+1)             |  
        # w83 = NN_{38}(x)*-1/2(y-1) * (1-x) * (x+1)                     |
        # w82 = NN_{28}(y)*-1/2(x-1) * (1-y) * (y+1)                     |


        w678  = ws['u678']  * ( (x[...,0] + 1) * (x[...,1] + 1) )[...,None]**alpha 
        w3478 = ws['u3478'] * ( (x[...,0] + 1) * (1 - x[...,1]) )[...,None]**alpha
        w238  = ws['u238']  * ( (1 - x[...,0]) * (1 - x[...,1]) )[...,None]**alpha 
        w1268 = ws['u1268'] * ( (1 - x[...,0]) * (x[...,1] + 1) )[...,None]**alpha

        # Function coinciding on the three subdomains
        #------------------------------------------------------------------------------#
        # w_678  = u_{678}  * ((x+1)*(y+1))^2   | 
        # w_3478 = u_{3478} * ((x+1)*(1-y))^2   |  
        # w_238  = u_{238}  * ((1-x)*(1-y))^2   | 
        # w_1268 = u_{1268} * ((1-x)*(y+1))^2   |  

        w =  w82 + w87 + w86 + w83 + w1268 + w238 + w678 + w3478
        return u * v + w
        

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
        grad5 = src.operators.gradient(lambda x : self.solution5(ws,x))(points['ys5'])[...,0,:]
        grad6 = src.operators.gradient(lambda x : self.solution6(ws,x))(points['ys6'])[...,0,:]
        grad7 = src.operators.gradient(lambda x : self.solution7(ws,x))(points['ys7'])[...,0,:]
        grad8 = src.operators.gradient(lambda x : self.solution8(ws,x))(points['ys8'])[...,0,:]
        
        #bi1 = jnp.einsum('mi,mij,mj->m',grad1,points['K1'],grad1)
        #bi2 = jnp.einsum('mi,mij,mj->m',grad2,points['K2'],grad2)
        #bi3 = jnp.einsum('mi,mij,mj->m',grad3,points['K3'],grad3)
        #bi4 = jnp.einsum('mi,mij,mj->m',grad4,points['K4'],grad4)

        #lpde1 = 0.5*(self.mu0)*jnp.dot(self.nu_model(bi1)*bi1, points['ws1']) 
        #lpde2 = 0.5*(self.mu0)*jnp.dot(self.nu_model(bi2)*bi2, points['ws2']) 
        #lpde3 = 0.5*(self.mu0)*jnp.dot(self.nu_model(bi3)*bi3, points['ws3']) 
        #lpde4 = 0.5*(self.mu0)*jnp.dot(self.nu_model(bi4)*bi4, points['ws4']) 


        lpde1 = 0.5 * 1/(self.mur*self.mu0) * jnp.dot(jnp.einsum('mi,mij,mj->m',grad1,points['K1'],grad1), points['ws1'])  
        lpde2 = 0.5 * 1/(self.mur*self.mu0) * jnp.dot(jnp.einsum('mi,mij,mj->m',grad2,points['K2'],grad2), points['ws2'])  
        lpde3 = 0.5 * 1/(self.mur*self.mu0) * jnp.dot(jnp.einsum('mi,mij,mj->m',grad3,points['K3'],grad3), points['ws3'])  
        lpde4 = 0.5 * 1/(self.mur*self.mu0) * jnp.dot(jnp.einsum('mi,mij,mj->m',grad4,points['K4'],grad4), points['ws4'])

        lpde5 = 0.5 * 1/self.mu0 * jnp.dot(jnp.einsum('mi,mij,mj->m',grad5,points['K5'],grad5), points['ws5'])  
        lpde6 = 0.5 * 1/self.mu0 * jnp.dot(jnp.einsum('mi,mij,mj->m',grad6,points['K6'],grad6), points['ws6'])  
        lpde7 = 0.5 * 1/self.mu0 * jnp.dot(jnp.einsum('mi,mij,mj->m',grad7,points['K7'],grad7), points['ws7'])  
        lpde8 = 0.5 * 1/self.mu0 * jnp.dot(jnp.einsum('mi,mij,mj->m',grad8,points['K8'],grad8), points['ws8'])  \
                - jnp.dot(self.J0*self.solution8(ws,points['ys8']).flatten()*points['omega8'] ,points['ws8'])

        return lpde1+lpde2+lpde3+lpde4+lpde5+lpde6+lpde7+lpde8

    def loss(self, ws, pts):
        lpde = self.loss_pde(ws, pts)
        return lpde
    

rnd_key = jax.random.PRNGKey(1235)
geoms = [iron_pole, iron_yoke, iron_yoke_r_mid, iron_yoke_r_low, air_1, air_2, air_3, current]
model = Model(rnd_key)
w0 = model.init_unravel()
weights = model.weights 
#plot_solution(rnd_key, model, weights)
#plt.show()


opt_type = 'ADAM'
batch_size = 8000
stepsize = 0.0001
n_epochs = 500

get_compiled = jax.jit(lambda key: model.get_points_MC(batch_size, key))
opt_init, opt_update, get_params = optimizers.adamax(step_size=stepsize)
opt_state = opt_init(weights)
params = get_params(opt_state)
evaluate_error(model, params)
print(params['u567'], params['u1268'], params['u3478'], params['u3478'], params['u678']) 
loss_grad = jax.jit(lambda ws, pts: (model.loss(ws, pts), jax.grad(model.loss)(ws, pts)))

key = jax.random.PRNGKey(np.random.randint(902763085663))
points = model.get_points_MC(batch_size, key)
def step(params, opt_state, key):
    # points = model.get_points_MC(batch_size, key)
    loss, grads = loss_grad(params, points)
    opt_state = opt_update(0, grads, opt_state)
    params = get_params(opt_state)
    return params, opt_state, loss


step_compiled = jax.jit(step)
step_compiled(params, opt_state, rnd_key)

tme = datetime.datetime.now()
for k in range(n_epochs):    
    params, opt_state, loss = step_compiled(params, opt_state, key)
    print(params['u567'], params['u1268'], params['u3478'], params['u3478'], params['u678']) 
    print('Epoch %d/%d - loss value %e'%(k+1, n_epochs, loss))
# update params

tme = datetime.datetime.now() - tme
print('Elapsed time ', tme)
save_models(params, '/home/mvt/iga_pinns/parameters/quad/')
print('Erfolgreich gespeichert!!')
evaluate_error(model, params)
