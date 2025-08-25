import os
N_CPU = 3
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={N_CPU}'

import jax
import jax.numpy as jnp
from dccxjax.jax_utils import *

x = jnp.arange(6*7*10).reshape((6,7,10))

SUM_AXIS = 1
BATCH_SIZE = N_CPU

print(jnp.sum(x, axis=[0,1,2].remove(SUM_AXIS)))
print(jax.vmap(jnp.sum, in_axes=SUM_AXIS, out_axes=0)(x))
print(batched_vmap(jnp.sum, BATCH_SIZE, in_axes=SUM_AXIS, out_axes=0)(x))

print(pmap_vmap(jnp.sum, "", BATCH_SIZE, in_axes=SUM_AXIS, out_axes=0)(x))

