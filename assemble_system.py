import jax
import jax.numpy as jnp
import numpy as np
import src
import src.bspline
from src.geometry import PatchNURBS
import matplotlib.pyplot as plt
from copy import copy
import scipy

def mke_circle(R, r):
   rmid = (R+r)/2
   controls = np.array([[[R,0],[r,0]] , [[R,R],[r,r]]  , [[0,R],[0,r]]]) 
   controls = np.array([[[R,0],[rmid,0],[r,0]] , [[R,R],[rmid,rmid], [r,r]]  , [[0,R],[0,rmid],[0,r]]]) 
   weights = jnp.array([[1,1,1],[1/np.sqrt(2),1/np.sqrt(2),1/np.sqrt(2)],[1,1,1]])
   return controls, weights

def preprocess_inputs(input_x1, input_x2):
   inputs_y = [input_x1, input_x2]
   y = np.meshgrid(*inputs_y)
   y = np.concatenate(tuple([k.flatten()[:,None] for k in y]),-1)
   return y

def mke_2d_NURBs(degrees, knots, controls, weights):
   key = jax.random.PRNGKey(0)
   spline_basis1 = src.bspline.BSplineBasisJAX(knots[0],degrees[0])
   spline_basis2 = src.bspline.BSplineBasisJAX(knots[1],degrees[1])
   basis = [spline_basis1, spline_basis2]
   my_nurbs_patch = PatchNURBS(basis, controls, weights, key)
   return basis, my_nurbs_patch

def mke_2d_quadrature(nurbs_patch, N):
   Knots = [(np.polynomial.legendre.leggauss(N)[0]+1)*0.5*(nurbs_patch.bounds[i][1]-nurbs_patch.bounds[i][0])+nurbs_patch.bounds[i][0] for i in range(nurbs_patch.d)]
   Ws = [np.polynomial.legendre.leggauss(N)[1]*0.5*(nurbs_patch.bounds[i][1]-nurbs_patch.bounds[i][0]) for i in range(nurbs_patch.d)]
   Knots = np.meshgrid(*Knots)
   ys = np.concatenate(tuple([k.flatten()[:,None] for k in Knots]),-1)
   Weights = Ws[0]
   for i in range(1,my_nurbs_patch.d):
        Weights = np.kron(Weights, Ws[i])
   return Weights, ys

def mke_ten_prod_derivative(b_spl_der, b_spl, dims):
   spl_der_x = []
   spl_der_y = []
   dim_x = dims[0]
   dim_y = dims[1]
   for i in range(dim_x):
       for j in range(dim_y):
           spl_der_x.append(b_spl_der[0][i]*b_spl[1][j])
           spl_der_y.append(b_spl[0][i]*b_spl_der[1][j])
   return spl_der_x, spl_der_y


def assemble_stiffness(b_spl, b_spl_der, dims, diff):
   dim_mat = dims[0]*dims[1]
   A = np.zeros((dim_mat,dim_mat))
   spl_der_x, spl_der_y = mke_ten_prod_derivative(b_spl_der, b_spl, dims) 
   
   for i in range(dim_mat):
       for j in range(dim_mat):
           integrand = spl_der_x[i]*spl_der_x[j] + spl_der_y[i]*spl_der_y[j]
           integrand = np.sum(diff*integrand)
           A[i][j] = integrand
   return A

def mke_rhs(dim_x, dim_y, b_spl, diff):
   b = []
   for i in range(dim_x):
       for j in range(dim_y):
           integrand = b_spl[0][i]*b_spl[1][j]
           integrand = np.sum(diff*integrand)
           b.append(integrand)
   b = np.array(b)
   return b


def mke_sampling_weights(nurbs_patch, ys, Weights):
    DGys = nurbs_patch._eval_omega(ys)
    diff = np.abs(DGys[:,0,0]*DGys[:,1,1] -  DGys[:,0,1]*DGys[:,1,0])*Weights
    return diff

def rotate_controls(controls, rotation_matrix): # currently specific to problem
   controls_int = np.reshape(controls, (9,2))
   controls_int = [np.matmul(rotation_matrix, controls_int[j,:]) for j in range(9)]
   controls_int = np.reshape(controls_int, (3,3,2)) 
   return controls_int

def apply_bcs(idxs_bc, A, b):
    b[idxs_bc] = 0
    insert_matrix = np.zeros((idxs_bc.shape[0],A.shape[1]))
    insert_matrix[range(idxs_bc.shape[0]), idxs_bc] = 1
    A[idxs_bc,:] = insert_matrix
    return A,b

def solve_system(degrees, idxs_bc, b_spl, b_spl_der, diff):
   dim_x = degrees[0]+1
   dim_y = degrees[1]+1
   A_neum = assemble_stiffness(b_spl, b_spl_der, [dim_x, dim_y], diff) 
   b_neum = mke_rhs(dim_x, dim_y, b_spl, diff)
   A, b = apply_bcs(idxs_bc, copy(A_neum), copy(b_neum))
   sol = scipy.linalg.solve(A,b, 'sym')
   return A_neum, b_neum, sol

def plot_solution(degrees, nurbs_patch, sol):
   dim_x = degrees[0]+1
   dim_y = degrees[1]+1

   xx, yy = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
   x_input = xx.flatten()
   y_input = yy.flatten()
   ys = np.array([x_input, y_input]).T
   
   pts = nurbs_patch.__call__(ys)
   ppx = np.reshape(pts[:,0],(50,50))
   ppy = np.reshape(pts[:,1],(50,50))
   b_spl = [basis[i].__call__(ys[:,i]) for i in range(my_nurbs_patch.d)]
   l=0
   eval_sum = 0
   for i in range(dim_x):
       for j in range(dim_y):
           curr_eval = sol[l]*b_spl[0][i]*b_spl[1][j]
           eval_sum = eval_sum + curr_eval
           l=l+1

   eval_sum = np.reshape(eval_sum, (50,50))
   plt.contourf(ppx,ppy,eval_sum, levels = 100)



if __name__ == '__main__':
   R=1
   r=0.2
   N=32

   degrees = [2,2]
   knots = [np.linspace(0,1,2), np.linspace(0,1,2)]


   controls, weights = mke_circle(R, r)
   rot_matrix = np.array([[0,1],[-1,0]])
   rot_controls = rotate_controls(controls, rot_matrix)
   idxs_bc = np.array([0,1,2,3,5,6,7,8])
   
   basis, my_nurbs_patch = mke_2d_NURBs(degrees, knots, controls, weights)
   _, rot_nurbs_patch = mke_2d_NURBs(degrees, knots, rot_controls, weights)
   
   weights, ys = mke_2d_quadrature(my_nurbs_patch, N)
   rot_weights, _ = mke_2d_quadrature(rot_nurbs_patch, N)

   b_spl = [basis[i].__call__(ys[:,i]) for i in range(my_nurbs_patch.d)]
   b_spl_der = [basis[i]._eval_basis_derivative(ys[:,i]) for i in range(my_nurbs_patch.d)]


   
   diff = mke_sampling_weights(my_nurbs_patch, ys, weights)
   rot_diff = mke_sampling_weights(rot_nurbs_patch, ys, rot_weights)

   A_neum, b_neum, sol = solve_system(degrees, idxs_bc, b_spl, b_spl_der, diff)
   A_neum_rot, b_neum_rot, sol_rot = solve_system(degrees, idxs_bc, b_spl, b_spl_der, rot_diff)
   print('Rhs: ', b_neum, b_neum_rot)

   A1_inner = A_neum_rot[0:6,0:6] # get the first six nodes from the Neumann system 
   A1_inner_interface = A_neum_rot[0:6,6:None] # inner nodes <-> interface 
   zero_insert = np.zeros_like((A1_inner))

   A2_inner = A_neum[3:None,3:None]
   A2_inner_interface = A_neum[3:None, 0:3]

   A_interface = (A_neum_rot[6:None, 6:None] + A_neum[0:3, 0:3])
   
   A1 = np.concatenate((A1_inner, zero_insert, A1_inner_interface), axis = 1)
   A2 = np.concatenate((zero_insert, A2_inner, A2_inner_interface), axis = 1)
   A3 = np.concatenate((A1_inner_interface.T, A2_inner_interface.T, A_interface), axis = 1)

   A_global = np.concatenate((A1,A2,A3)) 
   b_global = np.concatenate((b_neum_rot[0:6], b_neum[3:None], b_neum_rot[-3:None] + b_neum[0:3]))
   print('Verify rhs: ', np.sum(b_global),'==', 0.5*np.pi*(1-0.2**2))

   idx_bcs_global = np.array([0,1,2,3,5,6,8,9,10,11,12,14])
   apply_bcs(idx_bcs_global, A_global, b_global)

   print('The global right hand side is ', b_global)
   sol_global = scipy.linalg.solve(A_global, b_global, 'sym')
   print('Solution ', sol_global)
   sol_rot = np.concatenate((sol_global[0:6], sol_global[-3:None]))
   sol = np.concatenate((sol_global[-3:None], sol_global[-9:-3]))

   plot_solution(degrees, my_nurbs_patch, sol)
   plot_solution(degrees, rot_nurbs_patch, sol_rot)
   plt.colorbar()
   plt.show()









