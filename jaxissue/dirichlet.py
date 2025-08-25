import os
N_CPU = 10
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={N_CPU}'

import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.mesh_utils import create_device_mesh

def f(key):
    return jax.random.dirichlet(key, jnp.ones(3, float))

keys = jax.random.split(jax.random.key(0), 10)

# mesh = Mesh(create_device_mesh((N_CPU,)), "x")
# f_smap = shard_map.shard_map(jax.vmap(f), mesh=mesh, in_specs=P("x"), out_specs=P("x"))
# print(f_smap(keys))

print(jax.pmap(f)(keys))


#   zero = core.pvary(zero, tuple(core.typeof(alpha).vma))
#   one = core.pvary(one, tuple(core.typeof(alpha).vma))
#   minus_one = core.pvary(minus_one, tuple(core.typeof(alpha).vma))
#   two = core.pvary(two, tuple(core.typeof(alpha).vma))