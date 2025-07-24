N_CPU = 10

import os
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={N_CPU}'

import jax
# jax.config.update("jax_explain_cache_misses", True)

import jax.numpy as jnp
from dccxjax.all import *
import dccxjax.distributions as dist
from dccxjax.infer.mcmc.mcmc_core import get_mcmc_kernel, CarryStats, map_carry_stats, MCMCState
from dccxjax.infer.gibbs_model import GibbsModel
import time

# from jax._src.mesh_utils import create_device_mesh
# from jax import shard_map

from jax.experimental.mesh_utils import create_device_mesh
from jax.experimental.shard_map import shard_map # uses jax.shard_map but has @api_boundary


from jax.sharding import Mesh, PartitionSpec as P

import logging
setup_logging(logging.WARN)

def normal():
    x = sample("x", dist.Normal(0, 1))
    sample("y", dist.Normal(x, 1.), observed=1.)

m: Model = model(normal)()
slp = SLP_from_branchless_model(m)

regime =  MCMCStep(SingleVariable("x"), RW(gaussian_random_walk(1.)))
return_map = lambda x: x.position

fkernel, init_carry_stat_names = get_mcmc_kernel(slp, regime, vectorisation="vmap", return_map=return_map)

# slower
okernel, init_carry_stat_names = get_mcmc_kernel(slp, regime, vectorisation="none", return_map=return_map)
vkernel = vectorise_kernel_over_chains(okernel)

skernel, init_carry_stat_names = get_mcmc_kernel(slp, regime, vectorisation="smap", return_map=return_map)

# sskernel, init_carry_stat_names = get_mcmc_kernel(slp, regime, vectorised=3, return_map=return_map)

n_chains = 100
n_samples_per_chain = 1_000
assert n_chains % N_CPU == 0


keys = jax.random.split(jax.random.key(0), (n_samples_per_chain,))


init_positions = broadcast_jaxtree(slp.decision_representative, (n_chains,))
log_prob = jax.vmap(slp.log_prob, in_axes=(0,None,None))(init_positions, jnp.array(1.,float), dict())

infos = None

carry_stats = CarryStats(position=init_positions, log_prob=log_prob)

carry_stats = jax.vmap(map_carry_stats, in_axes=(0,None,None,None,None))(carry_stats, GibbsModel(slp, AllVariables()), jnp.array(1.,float), dict(), init_carry_stat_names)

initial_states = MCMCState(jnp.array(0, int), jnp.array(1.,float), dict(), init_positions, log_prob, carry_stats, infos)

# expected shape = (n_samples_per_chain,n_chains,...)

experiments = (
    # "fkernel",
    # "vkernel",
    # "skernel",
    # "okernel",
    "pmap",
    # "shard_map",
    # "shard_map okernel",
    # "shard_map vkernel",
    # "shard_map fkernel",
    # "smap",
    # "smap fkernel",
)

# "skernel", "shard_map fkernel" and "smap fkernel" seem equivalent ~8.5s
# "smap" slighly slower? ~9.3s
# "pmap" ~9.3s
# "shard_map okernel" ~11.5s
# "shard_map vkernel" ~10s
# "shard_map" ~10s


if "fkernel" in experiments:
    t0 = time.monotonic()
    last_state, res = jax.lax.scan(fkernel, initial_states, keys)
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape) # always puts chain at axis 1
    t1 = time.monotonic()
    print(f"fkernel {t1-t0:.3f}")

if "vkernel" in experiments:
    t0 = time.monotonic()
    last_state, res = jax.lax.scan(vkernel, initial_states, keys) # always puts chain at axis 1
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape)
    t1 = time.monotonic()
    print(f"vkernel {t1-t0:.3f}")

if "skernel" in experiments:
    t0 = time.monotonic()
    last_state, res = jax.lax.scan(skernel, initial_states, keys) # always puts chain at axis 1
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape)
    t1 = time.monotonic()
    print(f"skernel {t1-t0:.3f}")

@jax.jit
def chain(initial_state, keys):
    # print(initial_state, keys)
    return jax.lax.scan(okernel, initial_state, keys)
_axes = MCMCState(None,None,None,0,0,0,0) # type: ignore

CHAIN_AXIS = 1

if CHAIN_AXIS == 0:
    two_d_keys = jax.random.split(jax.random.key(0), (n_chains,n_samples_per_chain))
else:
    two_d_keys = jax.random.split(jax.random.key(0), (n_samples_per_chain,n_chains))

if "okernel" in experiments:
    t0 = time.monotonic()
    last_state, res = jax.vmap(chain, in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS))(initial_states, two_d_keys)
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape)
    t1 = time.monotonic()
    print(f"okernel {t1-t0:.3f}")

if "pmap" in experiments:
    if n_chains == N_CPU:
        t0 = time.monotonic()
        last_state, res = jax.pmap(chain, in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS))(initial_states, two_d_keys)
        print(res["x"])
        print(res["x"].shape, last_state.position["x"].shape)
        t1 = time.monotonic()
        print(f"pmap {t1-t0:.3f}")
        
        # does not work
        # t0 = time.monotonic()
        # last_state, res = jax.lax.scan(jax.pmap(okernel, in_axes=axes, out_axes=axes), initial_states, two_d_keys)
        # print(res["x"])
        # print(res["x"].shape)
        # t1 = time.monotonic()
        # print(f"pkernel {t1-t0:.3f}")
    else:
        # print("in", jax.tree.map(lambda v: v.shape, (initial_states, two_d_keys)))
        # last_state, res = jax.vmap(chain, in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS))(initial_states, two_d_keys)
        # print("out", res["x"].shape, last_state.position["x"].shape)
        
        three_d_key_shape = (N_CPU, n_chains // N_CPU, n_samples_per_chain) if CHAIN_AXIS == 0 else (n_samples_per_chain, N_CPU, n_chains // N_CPU)
        three_d_keys = two_d_keys.reshape(three_d_key_shape)
        vinitial_states = jax.tree.map(lambda v: v.reshape((N_CPU, n_chains // N_CPU) + v.shape[1:]) if len(v.shape) > 0 else v, initial_states)
        
        print("in", jax.tree.map(lambda v: v.shape, (vinitial_states, three_d_keys)))
        last_state, res = jax.pmap(
            jax.vmap(chain, in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS)),
            in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS), axis_name="i"
        )(vinitial_states, three_d_keys)
        print("out", res["x"].shape, last_state.position["x"].shape)
        print()
        
        pmap_vmap(chain, axis_name="i", batch_size=N_CPU, in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS))(initial_states, two_d_keys)

        exit()
        
        t0 = time.monotonic()
        three_d_key_shape = (N_CPU, n_chains // N_CPU, n_samples_per_chain) if CHAIN_AXIS == 0 else (n_samples_per_chain, N_CPU, n_chains // N_CPU)
        vinitial_states = jax.tree.map(lambda v: v.reshape((N_CPU, n_chains // N_CPU) + v.shape[1:]) if len(v.shape) > 0 else v, initial_states)
        last_state, res = jax.pmap(
            jax.vmap(chain, in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS)),
            in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS)
        )(vinitial_states, two_d_keys.reshape(three_d_key_shape))
        # print(res["x"])
        # print(res["x"].shape, last_state.position["x"].shape) # (n_samples_per_chain, N_CPU, n_chains // N_CPU, ...)
        out_shape = (n_chains, n_samples_per_chain) if CHAIN_AXIS == 0 else (n_samples_per_chain, n_chains,)
        res = jax.tree.map(lambda v: v.reshape(out_shape + v.shape[3:]) if len(v.shape) > 0 else v, res)
        last_state = jax.tree.map(lambda v: v.reshape((n_chains,) + v.shape[2:]) if len(v.shape) > 0 else v, last_state)
        print(res["x"])
        print(res["x"].shape, last_state.position["x"].shape)
        t1 = time.monotonic()
        print(f"pmap {t1-t0:.3f}")



# shard_map leaves the rank the same, whereas vmap would reduce the rank


device_mesh = create_device_mesh((N_CPU,))
mesh = Mesh(device_mesh, axis_names=("i"))
# print(device_mesh, mesh)

_specs = MCMCState(P(),P(),P(),P("i"),P("i"),P("i"),P("i")) # type: ignore
key_spec = P("i") if CHAIN_AXIS == 0 else P(None,"i")

if "shard_map" in experiments:
    t0 = time.monotonic()
    last_state, res = shard_map(jax.vmap(chain, in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS)), mesh=mesh, in_specs=(_specs,key_spec), out_specs=(_specs,key_spec))(initial_states, two_d_keys)
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape) # (n_chains, n_samples_per_chain) this is backwards
    t1 = time.monotonic()
    print(f"shard_map {t1-t0:.3f}")

# t0 = time.monotonic()
# last_state, res = shard_map(jax.vmap(chain, in_axes=(_axes,0), out_axes=(_axes,0)), mesh=mesh, in_specs=(_specs,P("i")), out_specs=(_specs,P("i")))(initial_states, jax.random.split(jax.random.key(0), (n_chains,n_samples_per_chain)))
# print(res["x"])
# print(res["x"].shape, last_state.position["x"].shape) # (n_chains, n_samples_per_chain) this is backwards
# t1 = time.monotonic()
# print(f"shard_map {t1-t0:.3f}")

if "shard_map okernel" in experiments:
    t0 = time.monotonic()
    scan_two_d_keys = jax.random.split(jax.random.key(0),(n_samples_per_chain,n_chains)).block_until_ready() # scan is always over axis 0
    last_state, res = jax.lax.scan(
        shard_map(jax.vmap(okernel, in_axes=(_axes,0), out_axes=(_axes,0)), mesh=mesh, in_specs=(_specs,P("i")), out_specs=(_specs,P("i"))),
        initial_states, scan_two_d_keys)
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape) # always puts chain at axis 1
    t1 = time.monotonic()
    print(f"shard_map okernel {t1-t0:.3f}")


if "shard_map vkernel" in experiments:
    t0 = time.monotonic()
    last_state, res = jax.lax.scan(shard_map(vkernel, mesh=mesh, in_specs=(_specs,P()), out_specs=(_specs,P("i"))), initial_states, keys)
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape) # always puts chain at axis 1
    t1 = time.monotonic()
    print(f"shard_map vkernel {t1-t0:.3f}")

if "shard_map fkernel" in experiments:
    t0 = time.monotonic()
    last_state, res = jax.lax.scan(shard_map(fkernel, mesh=mesh, in_specs=(_specs,P()), out_specs=(_specs,P("i"))), initial_states, keys)
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape) # always puts chain at axis 1
    t1 = time.monotonic()
    print(f"shard_map fkernel {t1-t0:.3f}")
    
# this does not work
# if "shard_map skernel" in experiments:
#     t0 = time.monotonic()
#     last_state, res = jax.lax.scan(shard_map(skernel, mesh=mesh, in_specs=(_specs,P()), out_specs=(_specs,P("i"))), initial_states, keys)
#     print(res["x"])
#     print(res["x"].shape, last_state.position["x"].shape) # always puts chain at axis 1
#     t1 = time.monotonic()
#     print(f"shard_map skernel {t1-t0:.3f}")


from jax._src.shard_map import smap

if "smap" in experiments:
    t0 = time.monotonic()
    with jax.sharding.use_mesh(mesh):
        # this does not work
        # last_state, res = smap(chain, in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS), axis_name="i")(initial_states, two_d_keys)
        # have to vmap to batch smap is just a shard_map wrapper
        last_state, res = smap(jax.vmap(chain, in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS)), in_axes=(_axes,CHAIN_AXIS), out_axes=(_axes,CHAIN_AXIS), axis_name="i")(initial_states, two_d_keys)
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape) # (n_chains, n_samples_per_chain) this is backwards
    t1 = time.monotonic()
    print(f"smap {t1-t0:.3f}")


if "smap fkernel" in experiments:
    t0 = time.monotonic()
    with jax.sharding.use_mesh(mesh):
        last_state, res = jax.lax.scan(smap(fkernel, in_axes=(_axes,None), out_axes=(_axes,0), axis_name="i"), initial_states, keys)
    print(res["x"])
    print(res["x"].shape, last_state.position["x"].shape) # always puts chain at axis 1
    t1 = time.monotonic()
    print(f"smap fkernel {t1-t0:.3f}")
