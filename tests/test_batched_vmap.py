import os
N_CPU = 3
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={N_CPU}'

import jax
import jax.numpy as jnp
from dccxjax.jax_utils import *

X = jnp.arange(6*7*10).reshape((6,7,10))

SUM_AXIS = 1
NUM_BATCHES = N_CPU

print(jnp.sum(X, axis=[0,1,2].remove(SUM_AXIS)))
print(jax.vmap(jnp.sum, in_axes=SUM_AXIS, out_axes=0)(X))
print(batched_vmap(jnp.sum, NUM_BATCHES, in_axes=SUM_AXIS, out_axes=0)(X))

print(pmap_vmap(jnp.sum, "", NUM_BATCHES, in_axes=SUM_AXIS, out_axes=0)(X))



print(batched_vmap(lambda x, y: jnp.sum(x) + y, NUM_BATCHES, in_axes=(SUM_AXIS,None), out_axes=0)(X, jnp.array(1,int)))
print(batched_vmap(lambda x, y: (jnp.sum(x), y), NUM_BATCHES, in_axes=(SUM_AXIS,None), out_axes=(0,None))(X, jnp.array(1,int)))