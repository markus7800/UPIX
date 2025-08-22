import jax
import jax.numpy as jnp
from dccxjax.jax_utils import *

x = jnp.arange(6*8*10).reshape((6,8,10))

SUM_AXIS = 1

print(jnp.sum(x, axis=[0,1,2].remove(SUM_AXIS)))
print(jax.vmap(jnp.sum, in_axes=SUM_AXIS, out_axes=0)(x))
print(batched_vmap(jnp.sum, 2, in_axes=SUM_AXIS, out_axes=0)(x))