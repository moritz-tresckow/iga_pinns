import jax
import jax.numpy as jnp
import numpy as np


x = np.array([1,2,3])
y = np.array([1,2,3])
a = np.arange(25).reshape(5,5)

total = np.einsum('ii', a)
print(total)