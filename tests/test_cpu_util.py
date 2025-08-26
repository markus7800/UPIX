import argparse
parser = argparse.ArgumentParser()
parser.add_argument("host_device_count", type=int)
args = parser.parse_args()

import os
os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={args.host_device_count}'
import time, os, jax, numpy as np, jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu') # insures we use the CPU

print(f"{jax.device_count()=}")
print(f"{jax.devices()=}")

def timer(name, f, x, shouldBlock=True):
   # warmup
   y = f(x).block_until_ready() if shouldBlock else f(x)
   # running the code
   start_wall = time.perf_counter()
   start_cpu = time.process_time()
   y = f(x).block_until_ready() if shouldBlock else f(x)
   end_wall = time.perf_counter()
   end_cpu = time.process_time()
   # computing the metric and displaying it
   wall_time = end_wall - start_wall
   cpu_time = end_cpu - start_cpu
   cpu_count = os.cpu_count()
   print(f"{name}: cpu usage {cpu_time/wall_time:.1f}/{cpu_count} wall_time:{wall_time:.1f}s")

# test functions
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, shape=(500_000_000,), dtype=float)
x_pmap = jax.random.normal(key, shape=(jax.device_count(), 500_000_000 // jax.device_count(),), dtype=float)
x_mat = jax.random.normal(key, shape=(10_000,10_000), dtype=float)
f_numpy = np.cos
f_vmap = jax.jit(jax.vmap(jnp.cos))
f_pmap = jax.pmap(jnp.cos)
f_dot = jax.jit(lambda x: jnp.dot(x,x.T)) # to show that JAX can indeed use all cores

timer('numpy', f_numpy, x, shouldBlock=False)
timer('vmap', f_vmap, x)
timer('pmap', f_pmap, x_pmap)
timer('dot', f_dot, x_mat)