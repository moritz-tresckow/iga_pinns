import jax
import jax.numpy as jnp
import numpy as np
import src
import src.bspline
import matplotlib.pyplot as plt
import unittest


class TestBSpline(unittest.TestCase):

    def test_spline(self):
        
        basis = src.bspline.BSplineBasis(np.linspace(0,1,7),2)



if __name__ == '__main__':
    
    basis = src.bspline.BSplineBasis(np.linspace(0,1,7),2)

    basis.spill_attributes() 

    x = np.linspace(0,1,1001)

    y = basis(x)



    plt.figure()
    plt.title('This is the numpy version')
    plt.plot(x,y.T)

    y = basis(x, derivative = True)
    plt.figure()
    plt.plot(x,y.T)
    
    
    basis = src.bspline.BSplineBasis(np.array([0,0.1,0.5,0.5,0.9,1]),2)
    
    y = basis(x)
    plt.figure()
    plt.plot(x,y.T)

    y = basis(x, derivative = True)
    plt.figure()
    plt.plot(x,y.T)
    
   
    
    
    # JAX Bspl

    basis = src.bspline.BSplineBasisJAX(np.linspace(0,1,7),2)
    #basisj = jax.jit(basis)

    
    x = np.linspace(0,1,1001)

    y = np.array(basis(x))
    plt.figure()
    plt.title('This is the jax version')
    plt.plot(x,y.T)

     
    y = basis(x, derivative = True)
    plt.figure()
    plt.plot(x,y.T)
    
    
    basis = src.bspline.BSplineBasisJAX(np.array([0,0.1,0.5,0.5,0.9,1]),2)
    
    y = basis(x)
    print(y.shape)
    
    plt.figure()
    plt.plot(x,y.T)

    y = basis(x, derivative = True)
    plt.figure()
    plt.plot(x,y.T)
    
    plt.show()
