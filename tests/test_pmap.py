N_CPU = 10

import os
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={N_CPU}'

import jax
# jax.config.update("jax_explain_cache_misses", True)

import jax.numpy as jnp
from dccxjax.all import *
import dccxjax.distributions as dist
import time

import logging
setup_logging(logging.WARN)

def normal():
    x = sample("x", dist.Normal(0, 1))
    sample("y", dist.Normal(x, 1.), observed=1.)

m: Model = model(normal)()
slp = SLP_from_branchless_model(m)

regime =  MCMCStep(SingleVariable("x"), RW(gaussian_random_walk(1.)))
return_map = lambda x: x.position

fkernel, init_carry_stat_names = mcmc.get_mcmc_kernel(slp, regime, vectorised=True, return_map=return_map)

# slower
okernel, init_carry_stat_names = mcmc.get_mcmc_kernel(slp, regime, vectorised=False, return_map=return_map)
vkernel = vectorise_kernel_over_chains(okernel)

n_chains = 100
n_samples_per_chain = 1_000_000
assert n_chains % N_CPU == 0


keys = jax.random.split(jax.random.key(0), (n_samples_per_chain,))


init_positions = broadcast_jaxtree(slp.decision_representative, (n_chains,))
log_prob = jax.vmap(slp.log_prob, in_axes=(0,None,None))(init_positions, jnp.array(1.,float), dict())

infos = None

carry_stats = mcmc.CarryStats(position=init_positions, log_prob=log_prob)

carry_stats = jax.vmap(mcmc.map_carry_stats, in_axes=(0,None,None,None,None))(carry_stats, gibbs_model.GibbsModel(slp, AllVariables()), jnp.array(1.,float), dict(), init_carry_stat_names)

initial_states = mcmc.MCMCState(jnp.array(0, int), jnp.array(1.,float), dict(), init_positions, log_prob, carry_stats, infos)

# expected shape = (n_samples_per_chain,n_chains,...)

t0 = time.monotonic()
last_state, res = jax.lax.scan(fkernel, initial_states, keys)
print(res["x"])
print(res["x"].shape, last_state.position["x"].shape)
t1 = time.monotonic()
print(f"fkernel {t1-t0:.3f}")

t0 = time.monotonic()
last_state, res = jax.lax.scan(vkernel, initial_states, keys)
print(res["x"])
print(res["x"].shape, last_state.position["x"].shape)
t1 = time.monotonic()
print(f"vkernel {t1-t0:.3f}")

# exit()
def chain(initial_state, keys):
    # print(initial_state, keys)
    return jax.lax.scan(okernel, initial_state, keys)
_axes = mcmc.MCMCState(None,None,None,0,0,0,0) # type: ignore
two_d_keys = jax.random.split(jax.random.key(0), (n_samples_per_chain,n_chains))

t0 = time.monotonic()
last_state, res = jax.vmap(chain, in_axes=(_axes,1), out_axes=(_axes,1))(initial_states, two_d_keys)
print(res["x"])
print(res["x"].shape, last_state.position["x"].shape)
t1 = time.monotonic()
print(f"okernel {t1-t0:.3f}")

if n_chains == N_CPU:
    t0 = time.monotonic()
    last_state, res = jax.pmap(chain, in_axes=(_axes,1), out_axes=(_axes,1))(initial_states, two_d_keys)
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape)
    t1 = time.monotonic()
    print(f"pkernel {t1-t0:.3f}")
    
    # does not work
    # t0 = time.monotonic()
    # last_state, res = jax.lax.scan(jax.pmap(okernel, in_axes=axes, out_axes=axes), initial_states, two_d_keys)
    # print(res["x"])
    # print(res["x"].shape)
    # t1 = time.monotonic()
    # print(f"pkernel {t1-t0:.3f}")
else:
    t0 = time.monotonic()
    vinitial_states = jax.tree.map(lambda v: v.reshape((N_CPU, n_chains // N_CPU) + v.shape[1:]) if len(v.shape) > 0 else v, initial_states)
    last_state, res = jax.pmap(
        jax.vmap(chain, in_axes=(_axes,1), out_axes=(_axes,1)),
        in_axes=(_axes,1), out_axes=(_axes,1)
    )(vinitial_states, two_d_keys.reshape((n_samples_per_chain, N_CPU, n_chains // N_CPU)))
    # print(res["x"])
    # print(res["x"].shape, last_state.position["x"].shape) # (n_samples_per_chain, N_CPU, n_chains // N_CPU, ...)
    res = jax.tree.map(lambda v: v.reshape((n_samples_per_chain, n_chains,) + v.shape[3:]) if len(v.shape) > 0 else v, res)
    last_state = jax.tree.map(lambda v: v.reshape((n_chains,) + v.shape[2:]) if len(v.shape) > 0 else v, last_state)
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape)
    t1 = time.monotonic()
    print(f"pkernel {t1-t0:.3f}")


from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

# shard_map leaves the rank the same, whereas vmap would reduce the rank


mesh = Mesh(mesh_utils.create_device_mesh((N_CPU,)), axis_names=("i"))
_specs = mcmc.MCMCState(P(),P(),P(),P("i"),P("i"),P("i"),P("i")) # type: ignore

t0 = time.monotonic()
last_state, res = shard_map(jax.vmap(chain, in_axes=(_axes,1), out_axes=(_axes,1)), mesh=mesh, in_specs=(_specs,P(None,"i")), out_specs=(_specs,P(None,"i")))(initial_states, two_d_keys)
print(res["x"])
print(res["x"].shape, last_state.position["x"].shape) # (n_chains, n_samples_per_chain) this is backwards
t1 = time.monotonic()
print(f"skernel {t1-t0:.3f}")

# t0 = time.monotonic()
# last_state, res = shard_map(jax.vmap(chain, in_axes=(_axes,0), out_axes=(_axes,0)), mesh=mesh, in_specs=(_specs,P("i")), out_specs=(_specs,P("i")))(initial_states, jax.random.split(jax.random.key(0), (n_chains,n_samples_per_chain)))
# print(res["x"])
# print(res["x"].shape, last_state.position["x"].shape) # (n_chains, n_samples_per_chain) this is backwards
# t1 = time.monotonic()
# print(f"skernel {t1-t0:.3f}")


# works but is a little bit slower
t0 = time.monotonic()
last_state, res = jax.lax.scan(shard_map(jax.vmap(okernel, in_axes=(_axes,0), out_axes=(_axes,0)), mesh=mesh, in_specs=(_specs,P("i")), out_specs=(_specs,P("i"))), initial_states, two_d_keys)
print(res["x"])
print(res["x"].shape)
t1 = time.monotonic()
print(f"sokernel {t1-t0:.3f}")


t0 = time.monotonic()
last_state, res = jax.lax.scan(shard_map(vkernel, mesh=mesh, in_specs=(_specs,P()), out_specs=(_specs,P("i"))), initial_states, keys)
print(res["x"])
print(res["x"].shape)
t1 = time.monotonic()
print(f"svkernel {t1-t0:.3f}")

t0 = time.monotonic()
last_state, res = jax.lax.scan(shard_map(fkernel, mesh=mesh, in_specs=(_specs,P()), out_specs=(_specs,P("i"))), initial_states, keys)
print(res["x"])
print(res["x"].shape)
t1 = time.monotonic()
print(f"sfkernel {t1-t0:.3f}")

# def step(carry, rng_key):
#     x = jax.random.normal(rng_key)
#     return carry + x, rng_key

# def vstep(carry, rng_key):
#     rng_keys = jax.random.split(rng_key, carry.shape[0])
#     x = jax.vmap(jax.random.normal)(rng_keys)
#     return carry + x, rng_keys
    
# n_chains = 2
# n_samples_per_chain = 3    
    
# keys = jax.random.split(jax.random.key(0), n_samples_per_chain)

# res = jax.lax.scan(jax.vmap(step), jnp.zeros((n_chains,),float), jax.random.split(keys,n_chains))
# print(res)

# res = jax.lax.scan(vstep, jnp.zeros((n_chains,),float), keys)
# print(res)


# n_chains = 3
# z = jax.random.key(0)
# jax.random.split(jax.random.split(z,2)[1], n_chains)
# Array((3,), dtype=key<fry>) overlaying:
# [[ 346279018  360566543]
#  [3968330031 3923691647]
#  [3152301319  792466127]]
# jax.vmap(lambda v: jax.random.split(v)[1])(jax.random.split(z,n_chains))
# Array((3,), dtype=key<fry>) overlaying:
# [[1353695780 2116000888]
#  [3968330031 3923691647]
#  [3777617834  145086855]]