import numpy as np
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


def write_data(params, path):
    target = flax.serialization.to_bytes(params)
    with open(path, "wb") as binary_file:
    # Write bytes to file
        binary_file.write(target)


def save_models(params, path):
    keys = list(params.keys())
    for i in keys:
        curr = params[i]
        if type(curr) != dict:
            save_vec = np.zeros((1,1))
            save_vec[0,0] = curr
            np.savetxt(path + i + '.csv', save_vec, delimiter = ',', header = '', comments = '')
        else:
            write_data(curr, path + i + '.txt')


class Res_MLP(nn.Module):
  feat: Sequence[int]
  
  @nn.compact
  def __call__(self, inputs):

    def act(x):
        return nn.tanh(x)

    x = inputs
    x = nn.Dense(self.feat[0])(x)
    x = nn.Dense(self.feat[1])(x)
    y = act(nn.Dense(self.feat[2])(x))
    y = act(nn.Dense(self.feat[3])(y))
    x = x + y
    x = nn.Dense(self.feat[-1])(x)
    return x


def model_init(feat):
    x = jnp.linspace(-1,1,feat[0])
    key1, _ = random.split(random.PRNGKey(0), 2)
    model = Res_MLP(feat)
    params = model.init(key1, x)
    return params, model


