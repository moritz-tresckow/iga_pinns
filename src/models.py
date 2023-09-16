import jax
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union)
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn

def load_data(model, path):
    with open(path, "rb") as binary_file:
        params = binary_file.read()
    params = flax.serialization.from_bytes(model, params)
    return params


def write_data(state, path):
    target = flax.serialization.to_bytes(state.params)
    with open(path, "wb") as binary_file:
    # Write bytes to file
        binary_file.write(target)



class Res_MLP(nn.Module):
  feat: Sequence[int]
  
  @nn.compact
  def __call__(self, inputs):

    def act(x):
        return nn.relu(x)

    x = inputs
    x = act(nn.Dense(self.feat[0])(x))
    x = act(nn.Dense(self.feat[1])(x))
    x = act(nn.Dense(self.feat[2])(x))
    x = nn.Dense(self.feat[-1])(x)
    return x


def model_init():
    x = jnp.linspace(-1,1,2)
    key1, _ = random.split(random.PRNGKey(0), 2)
    model = Res_MLP([2, 16, 16, 1])
    params = model.init(key1, x)
    return params, model

params, model = model_init()
print(type(model), type(params))
print(params)
u = model.apply(params, jnp.linspace(-1,1,2))
print(type(u))
