

import sys
sys.path.append("evaluation")
from parse_args import parse_args_and_setup # type: ignore
args = parse_args_and_setup()

import jax
from upix.all import *
import upix.distributions as dist
from setup_parallelisation import get_parallelisation_config # type: ignore

# jax.config.update("jax_explain_cache_misses", True)

import logging
setup_logging(logging.DEBUG)

def normal():
    x = sample("x", dist.Normal(0, 1))
    sample("y", dist.Normal(x, 1.), observed=1.)

m: Model = model(normal)()
slp = SLP_from_branchless_model(m)

regime =  MCMCStep(SingleVariable("x"), RW(gaussian_random_walk(1.)))

# regime =  MCMCStep(SingleVariable("x"), HMC(10, 0.1))

# regime =  MCMCStep(SingleVariable("x"), DHMC(10, 0.05, 0.15))

return_map = lambda x: x.position

# n_chains = 2**18
# n_samples_per_chain = 1_000


n_chains = 2**10
n_samples_per_chain = 100_000


mcmc_obj = MCMC(
    slp,
    regime,
    n_chains,
    parallelisation=get_parallelisation_config(args), # type:ignore
    return_map=return_map,
    collect_inference_info=True,
)


# last_state, result = timed(mcmc_obj.run)(jax.random.key(0),
#     StackedTrace(broadcast_jaxtree(slp.decision_representative, (n_chains,)), n_chains),
#     n_samples_per_chain=n_samples_per_chain)


import jax
import jax.numpy as jnp
from jax._src.shard_map import smap
from jax.sharding import Mesh
from jax.experimental.mesh_utils import create_device_mesh

I = jnp.array([0,1,2,3,5,6,7])

def f(xs, ys, key):
    def step(x, ix):
        x_new = x.at[ix].set(ys[ix])
        return x_new, None
    ixs = jax.random.permutation(key, I)
    # ixs = jnp.arange(len(ys))
    return jax.lax.scan(step, xs, ixs)

@jax.jit
def g(xs, ys, keys):
    return jax.lax.scan(lambda _, a: (None, f(*a)),  None, (xs,ys,keys))

T = 10
N = 10**6
xs = jax.random.normal(jax.random.key(0), (T, N,10))
ys = jax.random.normal(jax.random.key(1), (T, N,10))
keys = jax.random.split(jax.random.key(2), (T, N))

print(jax.vmap(g)(xs, ys, keys))

# print(jax.pmap(g)(xs,ys,keys))
    
# with jax.set_mesh(Mesh(create_device_mesh((jax.device_count(),)), axis_names=("i",))):
#     print(smap(jax.vmap(g), in_axes=0, out_axes=0, axis_name="i")(xs,ys,keys))

# with jax.set_mesh(Mesh(create_device_mesh((jax.device_count(),)), axis_names=("i",))):
#     x = jax.random.normal(jax.random.key(0), (100,))
#     y = jax.random.normal(jax.random.key(1), (100,))
#     print(smap(g, in_axes=0, out_axes=0, axis_name="i")(x,y))