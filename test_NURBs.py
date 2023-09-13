import jax
import jax.numpy as jnp
import numpy as np
import src
import src.bspline
from src.geometry import PatchNURBS
import matplotlib.pyplot as plt
import unittest


class TestBSpline(unittest.TestCase):

    def test_spline(self):
        
        basis = src.bspline.BSplineBasis(np.linspace(0,1,7),2)



if __name__ == '__main__':
   R=1
   r=0.2
   h=2

   key = jax.random.PRNGKey(0)
   knots = np.array([[[R,0],[r,0]] , [[R,R],[r,r]]  , [[0,R],[0,r]]]) 
   print(knots.shape)
   weights = jnp.array([[1,1],[1/np.sqrt(2),1/np.sqrt(2)],[1,1]])
   spline_basis1 = src.bspline.BSplineBasisJAX(np.linspace(0,1,2),2)
   spline_basis2 = src.bspline.BSplineBasisJAX(np.linspace(0,1,2),1)
   basis = [spline_basis1, spline_basis2]

   inputs_y = [np.linspace(0,1,20), np.linspace(0,1,20)]
   y = np.meshgrid(*inputs_y)
   y = np.concatenate(tuple([k.flatten()[:,None] for k in y]),-1)
   my_nurbs_patch = PatchNURBS(basis, knots, weights, key)
   pts = my_nurbs_patch.__call__(y)
   plt.scatter(pts[:,0], pts[:,1])
   plt.scatter(knots[:,:,0].flatten(), knots[:,:,1].flatten(), c = 'r')
   #plt.show()


   N=32
   Knots = [(np.polynomial.legendre.leggauss(N)[0]+1)*0.5*(my_nurbs_patch.bounds[i][1]-my_nurbs_patch.bounds[i][0])+my_nurbs_patch.bounds[i][0] for i in range(my_nurbs_patch.d)]
   Ws = [np.polynomial.legendre.leggauss(N)[1]*0.5*(my_nurbs_patch.bounds[i][1]-my_nurbs_patch.bounds[i][0]) for i in range(my_nurbs_patch.d)]
   Knots = np.meshgrid(*Knots)
   ys = np.concatenate(tuple([k.flatten()[:,None] for k in Knots]),-1)
   #plt.figure()
   #plt.scatter(ys[:,0], ys[:,1])
   
   print('The shape of ys is ', ys.shape)
   Weights = Ws[0]
   for i in range(1,my_nurbs_patch.d):
        Weights = np.kron(Weights, Ws[i])
   print(Weights.shape)

   DGys = my_nurbs_patch._eval_omega(ys)
   diff = np.abs(DGys[:,0,0]*DGys[:,1,1] -  DGys[:,0,1]*DGys[:,1,0])*Weights
   

   b_spl = [basis[i].__call__(ys[:,i]) for i in range(my_nurbs_patch.d)]
   b_spl_der = [basis[i]._eval_basis_derivative(ys[:,i]) for i in range(my_nurbs_patch.d)]
   print(b_spl[1][:].shape)
   A = np.zeros((6,6))

   spl_der_x = []
   spl_der_y = []
   for i in range(3):
       for j in range(2):
           spl_der_x.append(b_spl_der[0][i]*b_spl[1][j])
           spl_der_y.append(b_spl[0][i]*b_spl_der[1][j])
   

   for i in range(6):
       for j in range(6):
           integrand = spl_der_x[i]*spl_der_x[j] + spl_der_y[i]*spl_der_y[j]
           integrand = np.sum(diff*integrand)


           A[i][j] = integrand


   b = np.zeros((6))
   k=0
   for i in range(3):
       for j in range(2):
           integrand = b_spl[0][i]*b_spl[1][j]
           integrand = np.sum(diff*integrand)
           if i != 1:
               b[k] = 0
               zeros = np.zeros((6,))
               zeros[k] = 1
               print(zeros)
               A[k,:] = zeros
           else:
               b[k] = integrand
           k=k+1

   sol = np.linalg.solve(A,b) 
   print(sol)

   xx, yy = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
   x_input = xx.flatten()
   y_input = yy.flatten()
   ys = np.array([x_input, y_input]).T
   
   pts = my_nurbs_patch.__call__(ys)
   ppx = np.reshape(pts[:,0],(50,50))
   ppy = np.reshape(pts[:,1],(50,50))
   b_spl = [basis[i].__call__(ys[:,i]) for i in range(my_nurbs_patch.d)]
   l=0
   eval_sum = 0
   for i in range(3):
       for j in range(2):
           curr_eval = sol[l]*b_spl[0][i]*b_spl[1][j]
           print(curr_eval)
           eval_sum = eval_sum + curr_eval
           l=l+1

   eval_sum = np.reshape(eval_sum, (50,50))
   plt.figure()
   plt.contourf(ppx,ppy,eval_sum, levels = 100)
   plt.colorbar()
   plt.show()






