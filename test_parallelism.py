
from time import time

import os

os.environ["XLA_FLAGS"] = " ".join(
    ["--xla_force_host_platform_device_count=10"]
)
import jax
jax.config.update("jax_platform_name", "cpu")

cpus = jax.devices('cpu')
print(cpus)


N = 100_000_000
@jax.jit
def f(rng_key):
    def step(carry, key):
        return carry + jax.random.normal(key), None
    return jax.lax.scan(step, 0., jax.random.split(rng_key,N))[0]


# t0 = time()
# res = f(jax.random.PRNGKey(0))
# res.block_until_ready()
# t1 = time()
# print(f"Finished in {t1-t0:.3f}s")

t0 = time()
res = jax.vmap(f)(jax.random.split(jax.random.PRNGKey(0), (8,)))
res.block_until_ready()
print(res)
t1 = time()
print(f"Finished in {t1-t0:.3f}s")

t0 = time()
res = jax.pmap(f)(jax.random.split(jax.random.PRNGKey(0), (8,)))
res.block_until_ready()
print(res)
t1 = time()
print(f"Finished in {t1-t0:.3f}s")

t0 = time()
res = []
for i, key in enumerate(jax.random.split(jax.random.PRNGKey(0), (8,))):
    print(i)
    r = jax.jit(f, device=cpus[i])(jax.device_put(key, cpus[i]))
    res.append(r)
print(res)
t1 = time()
print(f"Finished in {t1-t0:.3f}s")

