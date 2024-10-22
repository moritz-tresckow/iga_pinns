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
def cal_rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def plot_knots(knots, c):
    plot_knots = np.reshape(knots, (knots.shape[0]*knots.shape[1],2))
    plt.scatter(plot_knots[:,0], plot_knots[:,1], c= c)


def sample_bnd(N):
    ys = np.linspace(-1,1,100)
    ys = ys[:,np.newaxis]
    one_vec = np.ones_like(ys)

    ys_top = np.concatenate((ys, one_vec), axis = 1)
    ys_bottom = np.concatenate((ys, -1 * one_vec), axis = 1)

    ys_left = np.concatenate((-1* one_vec, ys), axis = 1) 
    ys_right = np.concatenate((one_vec, ys), axis = 1) 
    bnd_pts = np.array([ys_top, ys_right, ys_bottom, ys_left])
    bnd_pts = np.reshape(bnd_pts, (bnd_pts.shape[0]*bnd_pts.shape[1], 2))
    return [ys_right, ys_bottom, ys_left, ys_top] 

#knots2 = np.array([[d4x + 5*delx, d4y-dely],[d4x + 4*delx, d4y + 2*dely],[d4x + delx, d4y + 2*dely] ,[d4x, d4y], [0.05, 0.00755176]])
def create_geometry(key, scale = 1):
    def get_poletip_knots(scale):
        scale = scale
        Dc = 5e-2                                                          
        hc = 7.55176e-3                                                           
        ri = 10e-3                                                           
        rm = (Dc*Dc+hc*hc-ri*ri)/(Dc*np.sqrt(2)+hc*np.sqrt(2)-2*ri)
        R = rm-ri
        O = np.array([rm/np.sqrt(2),rm/np.sqrt(2)])
        alpha1 = -np.pi*3/4       
        alpha2 = np.math.asin((hc-rm/np.sqrt(2))/R)
        alpha = np.abs(alpha2-alpha1)
    
        A = np.array([[O[0] - ri/np.sqrt(2), O[1] - ri/np.sqrt(2)], [O[0] - Dc, O[1] - hc]])
        b = np.array([[A[0,0]*ri/np.sqrt(2)+A[0,1]*ri/np.sqrt(2)],[A[1,0]*Dc+A[1,1]*hc]])
        C = np.linalg.solve(A,b)
        return C, alpha
    



    p1 = 0.12
    p2 = 0.1
    p3 = 0.05
    p4 = 0.01
    C, alpha = get_poletip_knots(scale)
    knots_outer = np.array([[p1, p2, p3, p4]]).T

    h = 0.03                    # Thickness of the pole tip

    d4x = p3 + h/np.sqrt(2)     # Calculate the first point parallel to the outer boundary. sqrt(2) for hypothenuse
    d4y = p3 - h/np.sqrt(2)

    delx = 0.005                # Increments in x and y direction
    dely = delx/2
    offset = 0.02               # Offset to shift the iron pole away from the origin 

    rotation_mat = cal_rotation_matrix(-np.pi/8)        # Rotation matrix for 22.5 degrees 
    rotation_mat2 = cal_rotation_matrix(-np.pi/32)      




    def mke_complete_ironyoke(knots_outer, offset, d4x, d4y, rotation_mat):
        knots_outer = np.concatenate((knots_outer, knots_outer), axis = 1)           # Create knots on outer iron yoke boundary defined by f(x)=x
        knot_bnd = np.matmul(rotation_mat, knots_outer[0,:])                         # Generate the final knot by rotating the last knot by 22.5 degrees
        knots_outer = np.concatenate((knot_bnd[np.newaxis,:], knots_outer))          # Add said knot to the knot vector
        

        knots_inner = np.array([[0.1162132, 0.0462868],[0.1112132, 0.0537868],[0.0962132, 0.0537868] ,[d4x, d4y], [0.05, 0.00755176]]) # Fix the inner nodes, leave the first node variable
        knots_middle = (knots_outer+knots_inner)/2                                         # Generate the knots between inner and outer nodes 
        knots_middle[-1,:] = C.flatten()                                             # Add nodes on the pole tips
        knots = np.concatenate((knots_outer[None,...],knots_middle[None,...],knots_inner[None,...]),0)
        knots = knots + offset
        
        weights = np.ones(knots.shape[:2])
        weights[1,-1] = np.sin((np.pi-alpha)/2)
        return knots, weights

    def split_iron_yoke(knots, weights): 
        weights_pole = weights[:,-2:None]
        weights_yoke = weights[:,0:-1]
        knots_pole = knots[:,-2:None,:] 

        knots_yoke = knots[:,0:-1,:]
        basisx = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
        basisy = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)
        iron_yoke = src.geometry.PatchNURBSParam([basisx, basisy], knots_yoke, weights_yoke, 0, 2, key)
        
        print(knots_pole, weights_pole)
        basisx = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
        basisy = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
        iron_pole = src.geometry.PatchNURBSParam([basisx, basisy], knots_pole, weights_pole, 0, 2, key) 
        return iron_pole, knots_pole, iron_yoke, knots_yoke

    def mke_right_yoke(knots_yoke):
        knots_top = knots_yoke[:,0,:]
        knots_bottom = np.matmul(rotation_mat, knots_top.T).T
        knots_bottom[:,1] = 0
        knots_middle = np.matmul(rotation_mat2, knots_top.T).T


        knots_iyr_mid = np.concatenate((knots_top[None,...],knots_middle[None,...]),0)
        knots_iyr_low = np.concatenate((knots_middle[None,...],knots_bottom[None,...]),0)

        weights_iyr_mid = np.ones(knots_iyr_mid.shape[:2])
        weights_iyr_low = np.ones(knots_iyr_low.shape[:2])
        
        basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
        basisy = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
        iron_yoke_r_mid = src.geometry.PatchNURBSParam([basisx, basisy], knots_iyr_mid, weights_iyr_mid , 0, 2, key) 
        iron_yoke_r_low = src.geometry.PatchNURBSParam([basisx, basisy], knots_iyr_low, weights_iyr_low , 0, 2, key) 
        return iron_yoke_r_mid, knots_iyr_mid, iron_yoke_r_low, knots_iyr_low

    def mke_air_domains(knots_pole, knots_iyr_mid, knots_iyr_low):
        a1 = knots_pole[0,1,:]
        a2 = knots_pole[1,-1,:]
        a3 = knots_pole[-1,-1,:]

        k1 = np.array([0.10, 0.035])
        k2 = knots_iyr_mid[1,-1,:]
        k3 = np.array([a3[0],0]) 
        k4 = knots_iyr_low[1,-1,:]
        f = lambda t : k1 + t*(k2-k1)
        f2 = lambda t : k3 + t*(k4-k3)

        knots_as = np.array([a1, a2, a3])
        
        knots = np.array([[[0,0],[a2[0],0],[a3[0],0]]])
        
        knots = np.concatenate((knots, knots_as[np.newaxis,:]))
        
        weights = np.ones(knots.shape[:2])
        weights[1,1] = np.sin((np.pi-alpha)/2)

        basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
        basisy = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
        air_1 = src.geometry.PatchNURBSParam([basisx, basisy], knots, weights, 0, 2, key)

        knots_bottom = np.array([k1, [a3[0],0]]) 
        knots_top = np.array([[d4x+offset,d4y+offset], a3])
        knots = np.concatenate((knots_top[None,...],knots_bottom[None,...]),0)
        weights = np.ones(knots.shape[:2])

        basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
        basisy = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
        air_2 = src.geometry.PatchNURBSParam([basisx, basisy], knots, weights, 0, 2, key)

        knots_air3_upper = np.array([f(1), f(0.75), f(0.25), f(0)])
        knots_air3_lower = np.array([f2(1), f2(0.75), f2(0.25), f2(0)])
        knots_air3 = np.concatenate((knots_air3_upper[None,...], knots_air3_lower[None,...]),0)
        weights = np.ones(knots_air3.shape[:2])

        basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
        basisy = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)
        air_3 = src.geometry.PatchNURBSParam([basisx, basisy], knots_air3, weights, 0, 2, key)
        return air_1, air_2, air_3, knots_air3 
    
    def mke_current_domain(knots_yoke, knots_air3):
        knots_curr_upper = knots_yoke[-1,:,:]
        knots_curr_lower =  knots_air3[0,:,:]
        knots_curr = np.concatenate((knots_curr_upper[None,...], knots_curr_lower[None,...]),0)
        weights = np.ones(knots_curr.shape[:2])
    
        basisx = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
        basisy = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),1)
        current = src.geometry.PatchNURBSParam([basisx, basisy], knots_curr, weights, 0, 2, key)
        return current
    
    knots, weights = mke_complete_ironyoke(knots_outer, offset, d4x, d4y, rotation_mat) 
    iron_pole, knots_pole, iron_yoke, knots_yoke = split_iron_yoke(knots, weights) 
    iron_yoke_r_mid, knots_iyr_mid, iron_yoke_r_low, knots_iyr_low = mke_right_yoke(knots_yoke)
    air_1, air_2, air_3, knots_air3 = mke_air_domains(knots_pole, knots_iyr_mid, knots_iyr_low)
    current = mke_current_domain(knots_yoke, knots_air3)
    
    return iron_pole, iron_yoke, iron_yoke_r_mid, iron_yoke_r_low, air_1, air_2, air_3, current





def mke_quadrupole_geo(key):
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
        return air1

    def mke_air2():
        #knots_lower =  np.array([a2, e0])
        # knots_lower =  np.array([e0, a2])
        knots_lower =  np.array([a2, b2])
        knots_lower = knots_lower[np.newaxis, :, :]
        #knots_upper =  np.array([b2, d0])
        # knots_upper =  np.array([d0, b2])
        knots_upper =  np.array([e0, d0])
        knots_upper = knots_upper[np.newaxis, :, :]

        knots = np.concatenate((knots_upper, knots_lower))
        weights = np.ones(knots.shape[:2])

        basisx = src.bspline.BSplineBasisJAX(np.array([-1,1]),1)
        basisy = src.bspline.BSplineBasisJAX(np.array([-1,1]),1)
        air2 = src.geometry.PatchNURBSParam([basisx, basisy], knots, weights, 0, 2, key)
        return air2


    def mke_iron_pole():
        knots_lower = np.array([b0, b1, b2])

        knots_lower = knots_lower[np.newaxis, :, :]

        knots_upper = np.array([c3, c0, d0])
        knots_upper = knots_upper[np.newaxis, :, :]
        knots_pole = np.concatenate((knots_upper, knots_lower))
        weights_pole = np.ones(knots_pole.shape[:2])
        weights_pole[1,1] = 0.7

        basisx = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
        basisy = src.bspline.BSplineBasisJAX(np.linspace(-1,1,2),1)
        iron_pole = src.geometry.PatchNURBSParam([basisy, basisx], knots_pole, weights_pole, 0, 2, key) 
        return iron_pole


    air1 = mke_air1()
    air2 = mke_air2()
    iron_pole = mke_iron_pole()
    return [iron_pole, air1, air2]

#create_geometry(rnd_key)
#geoms = mke_quadrupole_geo(rnd_key)

#bnd_samples = sample_bnd(1000)
#pts = [i.importance_sampling(1000)[0] for i in geoms]
#[plt.scatter(i[:,0], i[:,1]) for i in pts]

#cs = ['r', 'b', 'k', 'y']
#l = 1
#[plt.scatter(geoms[l].__call__(i)[:,0], geoms[l].__call__(i)[:,1], c = j) for i,j in zip(bnd_samples, cs)]
#plt.savefig('new_geo.png')
#exit()





















def mke_merged_patch(key):
    a = [0.0912132, 0.0487868]
    b = [0.1, 0.035]
    c = [0.0912132, 0]
    d = [0.1, 0]
    knots_upper_air2 = np.array([a,b]) 
    knots_lower_air2 = np.array([c,d]) 
    knots_upper_air2 = knots_upper_air2[np.newaxis, :, :]
    knots_lower_air2 = knots_lower_air2[np.newaxis, :, :]

    knots_mid_air2 = 0.5*(knots_lower_air2 + knots_upper_air2) 
    knots = np.concatenate((knots_lower_air2, knots_mid_air2, knots_upper_air2), axis = 0)
    weights = np.ones(knots.shape[:2])
    basis1 = src.bspline.BSplineBasisJAX(np.array([-1,0,1]),1)
    basis2 = src.bspline.BSplineBasisJAX(np.array([-1,1]),1)
    air_2 = src.geometry.PatchNURBSParam([basis1, basis2], knots, weights, 0, 2, key)



    a = [0.1, 0.035]
    b = [0.14205454, 0.05261638]
    c = [0.1, 0]
    d = [0.15121145, 0]

    knots_upper_air3 = np.array([d,c]) 
    knots_lower_air3 = np.array([b,a]) 

    knots_upper_air3 = knots_upper_air3[np.newaxis, :, :]
    knots_lower_air3 = knots_lower_air3[np.newaxis, :, :]
    knots_mid_air3 = 0.5*(knots_lower_air3 + knots_upper_air3) 
    knots = np.concatenate((knots_upper_air3, knots_mid_air3, knots_lower_air3), axis = 0)
    weights = np.ones(knots.shape[:2])
    basis1 = src.bspline.BSplineBasisJAX(np.array([-1,0,1]),1)
    basis2 = src.bspline.BSplineBasisJAX(np.array([-1,1]),1)
    air_3 = src.geometry.PatchNURBSParam([basis1, basis2], knots, weights, 0, 2, key)

    knots_upper_air1 = np.array([[0.03, 0.03], [0.045, 0.01], [0.07, 0.02755176], [0.07, 0.02755176], [0.0912132, 0.0487868]]) 
    knots_upper_air1 = knots_upper_air1[np.newaxis, :, :]
    knots_lower_air1 = np.array([[0, 0], [0.045, 0], [0.07, 0], [0.07, 0], [0.0912132, 0]]) 
    knots_lower_air1 = knots_lower_air1[np.newaxis, :, :]

    knots_mid_air1 = 0.5*(knots_lower_air1 + knots_upper_air1) 
    knots = np.concatenate((knots_lower_air1, knots_mid_air1, knots_upper_air1), axis = 0)
    weights = np.ones(knots.shape[:2])
    weights[2,1] = 0.5

    basis1 = src.bspline.BSplineBasisJAX(np.array([-1,1]),2)
    basis2 = src.bspline.BSplineBasisJAX(np.array([-1,-0.33,0.33,1]),2)
    air_1 = src.geometry.PatchNURBSParam([basis1, basis2], knots, weights, 0, 2, key)

    air_1_pts, diffs = air_1.importance_sampling(10000)
    air_2_pts, diffs = air_2.importance_sampling(5000)
    air_3_pts, diffs = air_3.importance_sampling(10000)
    plt.scatter(air_1_pts[:,0], air_1_pts[:,1], s = 0.1)
    plt.scatter(air_2_pts[:,0], air_2_pts[:,1], s = 0.1)
    plt.scatter(air_3_pts[:,0], air_3_pts[:,1], s = 0.1)
    plt.savefig('./cer.png')
    return air_1, air_2, air_3




def plot_geo_bnd(geom):
    #iron_pole, iron_yoke, iron_yoke_r_mid, iron_yoke_r_low, air_1, air_2, air_3, current  = create_geometry(rnd_key)
    print('Starting domain calculations')

    ys = np.linspace(-1,1,100)
    xx,yy = np.meshgrid(ys, ys)
    input_vec = np.concatenate((xx.flatten()[:,np.newaxis], yy.flatten()[:,np.newaxis]), axis = 1)
    # iron_pole_pts,_ = iron_pole.importance_sampling(1000)
    #iron_yoke_pts,_ = iron_yoke.importance_sampling(1000)
    #current_pts,_ = current.importance_sampling(1000)
    #air_3_pts,_ = air_3.importance_sampling(1000)
    #air_2_pts,_ = air_2.importance_sampling(1000)
    #air_1_pts,_ = air_1.importance_sampling(1000)
    #iron_yoke_r_mid_pts,_ = iron_yoke_r_mid.importance_sampling(300)
    #iron_yoke_r_low_pts,_ = iron_yoke_r_low.importance_sampling(1000)
    #geom_pts,_ = geom.importance_sampling(1000)
    
    iron_pole_pts = iron_pole.__call__(input_vec)
    iron_yoke_pts = iron_yoke.__call__(input_vec)
    current_pts = current.__call__(input_vec)
    air_3_pts = air_3.__call__(input_vec)
    air_2_pts = air_2.__call__(input_vec)
    air_1_pts = air_1.__call__(input_vec)
    iron_yoke_r_mid_pts = iron_yoke_r_mid.__call__(input_vec)
    iron_yoke_r_low_pts = iron_yoke_r_low.__call__(input_vec)

    pts = [iron_pole_pts, iron_yoke_pts, current_pts, air_3_pts, air_2_pts, air_1_pts, iron_yoke_r_mid_pts, iron_yoke_r_low_pts]
    #[plt.scatter(i[:,0], i[:,1]) for i in pts]
    #plt.show()
    [np.reshape(i, (100**2, 2)) for i in pts]
    pts = np.array(pts)
    pts = np.reshape(pts, (pts.shape[0]*pts.shape[1], 2))
    np.savetxt('./coordinates.csv', pts, delimiter = ',', comments = '')
    exit()
    print('Ending domain calculations')


    print('Starting boundary calculations')
    
    ys = jax.random.uniform(rnd_key, (1000, 2))
    ys = 2*ys - 1
    ys_right = ys.at[:,0].set(1)
    bnd_right = geom.__call__(ys_right)
    ys_top = ys.at[:,1].set(1)
    bnd_top = geom.__call__(ys_top) 
    ys_left = ys.at[:,0].set(-1)
    bnd_left = geom.__call__(ys_left) 
    ys_bottom = ys.at[:,1].set(-1)
    bnd_bottom = geom.__call__(ys_bottom)

    print('Ending boundary calculations')

    plt.figure()
    plt.scatter(iron_pole_pts[:,0], iron_pole_pts[:,1], s = 1, c='k')
    plt.scatter(iron_yoke_pts[:,0], iron_yoke_pts[:,1], s = 1, c='k')
    plt.scatter(current_pts[:,0], current_pts[:,1], s = 1, c='k')
    plt.scatter(air_3_pts[:,0], air_3_pts[:,1], s = 1, c='k')
    plt.scatter(air_2_pts[:,0], air_2_pts[:,1], s = 1, c='k')
    plt.scatter(air_1_pts[:,0], air_1_pts[:,1], s = 1, c='k')
    plt.scatter(iron_yoke_r_mid_pts[:,0], iron_yoke_r_mid_pts[:,1], s = 1, c='k')
    plt.scatter(iron_yoke_r_low_pts[:,0], iron_yoke_r_low_pts[:,1], s = 1, c='k')
    plt.scatter(geom_pts[:,0], geom_pts[:,1], s = 4, c='r')

    
    plt.scatter(bnd_right[:,0], bnd_right[:,1], c='y')
    plt.scatter(bnd_top[:,0], bnd_top[:,1], c='g')
    plt.scatter(bnd_left[:,0], bnd_left[:,1], c='b')
    plt.scatter(bnd_bottom[:,0], bnd_bottom[:,1], c='m')


    plt.figure()

    plt.scatter(ys_right[:,0], ys_right[:,1], c='y')
    plt.scatter(ys_top[:,0], ys_top[:,1], c='g')
    plt.scatter(ys_left[:,0], ys_left[:,1], c='b')
    plt.scatter(ys_bottom[:,0], ys_bottom[:,1], c='m')



# iron_pole, iron_yoke, iron_yoke_r_mid, iron_yoke_r_low, air_1, air_2, air_3, current  = create_geometry(rnd_key)
# plot_geo_bnd(air_3)
# plt.show()
# iron_pole, iron_yoke, iron_yoke_r_mid, iron_yoke_r_low, air_1, air_2, air_3, current  = create_geometry(rnd_key)


def plot_single_domain(model, weights):
    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
    sol = model.solution6(weights, ys)
    sol = np.abs(np.reshape(sol, (100, 100)))
    plt.contourf(x,y,sol, levels = 100)
    plt.colorbar()
    plt.show()

def plot_bndr(model, weights, geoms):
    def sample_bnd():
        ys = np.linspace(-1,1,1000)
        ys = ys[:,np.newaxis]
        one_vec = np.ones_like(ys)

        ys_top = np.concatenate((ys, one_vec), axis = 1)
        ys_bottom = np.concatenate((-1* ys, -1 * one_vec), axis = 1)

        ys_left = np.concatenate((-1 * one_vec, ys), axis = 1) 

        ys_right = np.concatenate((one_vec, -1 * ys), axis = 1) 

       
        return ys, [ys_top, ys_right, ys_bottom, ys_left] 

    pole_tip = geoms[0]
    iron_yoke = geoms[1]
    iy_right_middle = geoms[2]
    iy_right_lower = geoms[3]
    air1 = geoms[4]
    air2 = geoms[5]
    air3 = geoms[6]
    current = geoms[7]
    ys, samples = sample_bnd()
    # [plt.scatter(air1.__call__(i)[:,0], air1.__call__(i)[:,1], c = 'g') for i in samples]
    # [plt.scatter(air2.__call__(i)[:,0], air2.__call__(i)[:,1], c = 'b') for i in samples]
    # plt.scatter(air1.__call__(samples[0])[:,0], air1.__call__(samples[0])[:,1], c = 'k') 
    # plt.scatter(air2.__call__(samples[0])[:,0], air2.__call__(samples[0])[:,1], c = 'k') 
    # plt.scatter(p1[0,0], p1[0,1], c = 'r')
    # plt.scatter(p2[0,0], p2[0,1], c = 'r')
    # plt.show()
    # exit()



    vals1 = [model.solution1(weights, i) for i in samples]
    vals2 = [model.solution2(weights, i) for i in samples]
    vals3 = [model.solution3(weights, i) for i in samples]
    vals4 = [model.solution4(weights, i) for i in samples]
    vals5 = [model.solution5(weights, i) for i in samples]
    vals6 = [model.solution6(weights, i) for i in samples]
    vals7 = [model.solution7(weights, i) for i in samples]
    vals8 = [model.solution8(weights, i) for i in samples]
    
    left_brd = jnp.array([[-1,1]]) 
    right_brd = jnp.array([[1,1]]) 

    ys = np.linspace(-1,1,1000)
    ys = ys[:,np.newaxis]
    ones_vec = np.ones_like(ys)
    input_1 = np.concatenate((ones_vec, ys), axis = 1)
    input_2 = np.concatenate((-1 * ones_vec, ys), axis = 1)
    sol5 = model.solution5(weights, input_1)
    sol6 = model.solution6(weights, input_2)
    sol_5l = model.solution5(weights, right_brd)
    sol_6l = model.solution6(weights, left_brd)
    p1 = pole_tip.__call__(input_1)
    p2 = air2.__call__(input_2)
    print('The difference in points is ', np.abs(np.sum(p1 - p2)))
     
    plt.figure()
    plt.plot(ys, vals5[1], label = 'u51')
    plt.plot(ys, np.flip(vals1[0]), label = 'u15')
    plt.savefig('bndry_u51.png')
    plt.legend()


    plt.figure()
    plt.plot(ys, vals1[1], label = 'u16')
    plt.plot(ys,np.flip(vals6[3]), label = 'u61')
    plt.legend()
    plt.savefig('bndry_u61.png')


    plt.figure()
    plt.plot(ys, vals6[1], label = 'u67')
    plt.plot(ys, np.flip(vals7[0]), label = 'u76')
    plt.legend()
    plt.savefig('bndry_u67.png')
#
    plt.figure()
    plt.plot(ys, vals6[2], label = 'u68')
    plt.plot(ys, np.flip(vals8[0]), label = 'u86')
    plt.legend()
    plt.savefig('bndry_u68.png')
    #
    plt.figure()
    plt.plot(ys, vals1[2], label = 'u12')
    plt.plot(ys, np.flip(vals2[0]), label = 'u21')
    plt.legend()
    plt.savefig('bndry_u21.png')
#
#
    plt.figure()
    plt.plot(ys, vals2[1], label = 'u28')
    plt.plot(ys, np.flip(vals8[3]), label = 'u82')
    plt.legend()
    plt.savefig('bndry_u28.png')
#

#
    plt.figure()
    plt.plot(ys, vals3[0], label = 'u38')
    plt.plot(ys, np.flip(vals8[2]), label = 'u83')
    plt.legend()
    plt.savefig('bndry_u38.png')
#
    plt.figure()
    plt.plot(ys, vals3[1], label = 'u34')
    plt.plot(ys, np.flip(vals4[3]), label = 'u43')
    plt.legend()
    plt.savefig('bndry_u34.png')
#
    plt.figure()
    plt.plot(ys, vals3[3], label = 'u32')
    plt.plot(ys, np.flip(vals2[2]), label = 'u23')
    plt.legend()
    plt.savefig('bndry_u32.png')
    
    plt.figure()
    plt.plot(ys, vals5[0], label = 'u56')
    plt.plot(ys, np.flip(vals6[0]), label = 'u65')
    plt.legend()
    plt.savefig('bndry_u56.png')
 #
    plt.figure()
    plt.plot(ys, vals8[0], label = 'u86')
    plt.plot(ys, np.flip(vals6[2]), label = 'u68')
    plt.legend()
    plt.savefig('bndry_u86.png')
    
    plt.figure()
    plt.plot(ys, vals8[1], label = 'u87')
    plt.plot(ys, np.flip(vals7[3]), label = 'u78')
    plt.legend()
    plt.savefig('bndry_u78.png')
#
    plt.figure()
    plt.plot(ys, vals8[3], label = 'u82')
    plt.plot(ys, np.flip(vals2[1]), label = 'u28')
    plt.legend()
    plt.savefig('bndry_u82.png')
#
#
    plt.figure()
    plt.plot(ys, vals7[0], label = 'u76')
    plt.plot(ys, np.flip(vals6[1]), label = 'u67')
    plt.legend()
    plt.savefig('bndry_u76.png')
    
    plt.figure()
    plt.plot(ys, vals7[2], label = 'u74')
    plt.plot(ys, np.flip(vals4[0]), label = 'u47')
    plt.legend()
    plt.savefig('bndry_u74.png')




def plot_solution(rnd_key, model, params):
    iron_pole, iron_yoke, iron_yoke_r_mid, iron_yoke_r_low, air_1, air_2, air_3, current  = create_geometry(rnd_key)
    model.weights = params
    weights = params
    

    x,y = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
    ys = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),1)
    xy1 = iron_pole(ys)
    xy2 = iron_yoke(ys)
    xy3 = iron_yoke_r_mid(ys)
    xy4 = iron_yoke_r_low(ys)

    xy5 = air_1(ys)
    xy6 = air_2(ys)
    xy7 = air_3(ys)
    xy8 = current(ys)

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
    print(vmin, vmax)

    #vmin = min([u5.min(),u6.min(),u7.min()]) 
    #vmax = max([u5.max(),u6.max(),u7.max()])

    plt.figure(figsize = (20,12))
    ax = plt.gca()
    plt.contourf(xy1[:,0].reshape(x.shape), xy1[:,1].reshape(x.shape), u1, levels = 100, vmin = vmin, vmax = vmax)
    plt.contourf(xy2[:,0].reshape(x.shape), xy2[:,1].reshape(x.shape), u2, levels = 100, vmin = vmin, vmax = vmax)
    plt.contourf(xy3[:,0].reshape(x.shape), xy3[:,1].reshape(x.shape), u3, levels = 100, vmin = vmin, vmax = vmax)
    plt.contourf(xy4[:,0].reshape(x.shape), xy4[:,1].reshape(x.shape), u4, levels = 100, vmin = vmin, vmax = vmax)
    plt.contourf(xy5[:,0].reshape(x.shape), xy5[:,1].reshape(x.shape), u5, levels = 100, vmin = vmin, vmax = vmax)
    plt.contourf(xy6[:,0].reshape(x.shape), xy6[:,1].reshape(x.shape), u6, levels = 100, vmin = vmin, vmax = vmax)
    plt.contourf(xy7[:,0].reshape(x.shape), xy7[:,1].reshape(x.shape), u7, levels = 100, vmin = vmin, vmax = vmax)
    plt.contourf(xy8[:,0].reshape(x.shape), xy8[:,1].reshape(x.shape), u8, levels = 100, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.xlabel(r'$x_1$ [m]')
    plt.ylabel(r'$x_2$ [m]')
    plt.show()

"""
knots_bottom = np.array([[a3[0],0], [d4x+offset,0]])
knots_bottom = np.array([knots_bottom[0], 0.5*(knots_bottom[0] + knots_bottom[1]), knots_bottom[1]])
knots_middle = np.array([a3, [0.0912132, 0.0243934]] ) 
knots_middle = np.array([knots_middle[0], 0.5*(knots_middle[0] + knots_middle[1]), knots_middle[1]])
knots_air1 = np.array([[d4x+offset,d4y+offset], k1])
knots_air1 = np.array([knots_air1[0], 0.5*(knots_air1[0] + knots_air1[1]), knots_air1[1]])
"""
