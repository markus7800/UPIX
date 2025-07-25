import os
os.environ["XLA_FLAGS"] = " ".join(
    ["--xla_force_host_platform_device_count=10"]
)
import time

import jax
import jax.numpy as jnp
devices = jax.devices()
print(devices)
from jax.random import PRNGKey


def timer(name, f, x, shouldBlock=True):
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



# N = 10_000
# @jax.jit
# def g(rng_key):
#     X = jax.random.normal(rng_key, (N,N))
#     return (X @ X @ X @ X @ X).sum()

# timer("g", g, jax.random.key(0), True)

# def g_pmap(rng_key):
#    return jax.pmap(g, devices=[devices[0]])(jax.lax.broadcast(rng_key, (1,)))

# timer("g_pmap", g, jax.random.key(0), True)


@jax.jit
def step(carry, any):
  val, key = carry
  key1, key2 = jax.random.split(key)
  val = val + jax.random.normal(key1)
  return (val,key2), None
  
@jax.jit
def f(seed):
  print(f"Compile f {seed}")
  return jax.lax.scan(step, (jnp.array(0.,float), seed), length=10**7)[0][0]

# e = f.trace(jax.random.key(0)).lower().compile()
# print(e)
# from jax.export import export

# exported = export(f)(jax.random.key(0))
# print(exported)

# f = exported.call

timer("base", f, jax.device_put(key(0), devices[0]), shouldBlock=True)

def f2(x):
    x1, x2 = x
    r1 = f(x1)
    r2 = f(x2)
    print(f"{r1}, {r2}")


timer("async 1 device", f2, (jax.device_put(key(0), devices[0]),jax.device_put(key(1), devices[0])), shouldBlock=False)

# timer("async 2 devices", f2, (jax.device_put(key(0), devices[0]),jax.device_put(key(1), devices[1])), shouldBlock=False)

# async def f_async(x):
#    return f(x)

# import asyncio
# async def f2_async(x):
#     t0 = time.time()
#     x1, x2 = x
#     task1 = asyncio.create_task(f_async(x1))
#     task2 = asyncio.create_task(f_async(x2))
#     r1 = await task1
#     r2 = await task2
#     t1 = time.time()
#     print(f"{r1}, {r2} - {t1-t0:.3f}s")
# asyncio.run(f2_async((key(0),key(1))))

# import threading
# threads = []
# t = threading.Thread(target=f, args=(key(0),))
# threads.append(t)
# t = threading.Thread(target=f, args=(key(1),))
# threads.append(t)

# t0 = time.time()
# # Start each thread
# for t in threads:
#     t.start()

# # Wait for all threads to finish
# for t in threads:
#     t.join()
# t1 = time.time()
# print(f"{t1-t0:.3f}s")
exit()

def f_pmap(x):
   r = jax.pmap(f)(x)
   print(r)

timer("pmap", f_pmap, jnp.vstack((key(0), key(1))), shouldBlock=False)

def f_pmap_2(x):
   r = jax.pmap(f, devices=[devices[0],devices[1]])(x)
   print(r)

timer("pmap device", f_pmap_2, jnp.vstack((key(0), key(1))), shouldBlock=False)


def f3(x):
    x1, x2 = x
    r1 = jax.pmap(f, devices=[devices[0]])(jax.lax.broadcast(x1,(1,)))
    r2 = jax.pmap(f, devices=[devices[1]])(jax.lax.broadcast(x2,(1,)))
    print(f"{r1}, {r2}")

timer("pmap async", f3, (key(0), key(1)), shouldBlock=False)