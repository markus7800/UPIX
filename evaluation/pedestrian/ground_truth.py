import jax
import jax.numpy as jnp
from typing import NamedTuple
from time import time

class PedWhileState(NamedTuple):
    rng_key: jax.Array
    position: jax.Array
    distance: jax.Array
    t: jax.Array

def cond(state: PedWhileState):
    return (state.position > 0) & (state.distance < 10)

def body_fun(state: PedWhileState):
    new_rng_key, sample_key = jax.random.split(state.rng_key)
    step = jax.random.uniform(sample_key) * 2 - 1
    return PedWhileState(new_rng_key, state.position + step, state.distance + jax.lax.abs(step), state.t + 1)
    
@jax.jit
def pedestrian1(rng_key: jax.Array):
    # print(rng_key)
    rng_key, sample_key = jax.random.split(rng_key)
    start = jax.random.uniform(sample_key) * 3.
    lp = jax.lax.log(1./3.)

    end_state = jax.lax.while_loop(cond, body_fun, PedWhileState(rng_key, start, jnp.array(0.), jnp.array(0)))

    lp += end_state.t * jax.lax.log(1./2.)

    lp += jax.scipy.stats.norm.logpdf(1.1, end_state.distance, 0.1)

    return {"start": start, "lp": lp}

def pedestrian2(rng_key: jax.Array):
    rng_key, sample_key = jax.random.split(rng_key)
    start = jax.random.uniform(sample_key) * 3.
    lp = jax.lax.log(1./3.)
    position = start
    distance = 0
    t = 0
    while position > 0 and distance < 10:
        rng_key, sample_key = jax.random.split(rng_key)
        step = jax.random.uniform(sample_key) * 2. - 1.
        position += step
        distance += jax.lax.abs(step)
        # lp += jax.lax.log(1./2.)
        t += 1
    lp += t * jax.lax.log(1./2.)
    lp += jax.scipy.stats.norm.logpdf(1.1, distance, 0.1)

    return {"start": start, "lp": lp}


def pedestrian_batched(seed: jax.Array, N: int, M: int):
    rng_keys = jax.random.split(seed, (N, M))
    return jax.vmap(jax.vmap(pedestrian1))(rng_keys)

# pedestrian_batched(jax.random.PRNGKey(0), N, 1) == [pedestrian1(key) for key in jax.random.split(jax.random.PRNGKey(0), N)]

t0 = time()
result = pedestrian_batched(jax.random.PRNGKey(0), 1_000_000, 1) # not batching while loop is faster on CPU
result["start"].block_until_ready()
x = result["start"].reshape(-1)
lp = result["lp"].reshape(-1)
t1 = time()
print(f"Finished in {t1-t0:.3f}s")

# result = [pedestrian1(key) for key in jax.random.split(jax.random.PRNGKey(0), 10)]
# x = jnp.hstack([r["start"] for r in result])
# lp = jnp.hstack([r["lp"] for r in result])
# print(x)
# print(lp)

import matplotlib.pyplot as plt

weights = jax.lax.exp(lp - jax.scipy.special.logsumexp(lp))
print(weights.sum())


# plt.hist(x, weights=weights, density=True, bins=100)
# plt.show()

qs = jnp.linspace(0., 3., 100)


@jax.jit
def cdf(x, qs, weights):
    def _cdf(q):
        return jnp.where(x < q, weights, 0.).sum()
    return jax.lax.map(_cdf, qs)

def cdf_cruncher(qs, N):
    rng_keys = jax.random.split(jax.random.PRNGKey(0), N)
    c = jnp.zeros_like(qs)
    for (i,key) in enumerate(rng_keys):
        t0 = time()
        result = pedestrian_batched(key, 10_000_000, 1)
        x = result["start"].reshape(-1)
        lp = result["lp"].reshape(-1)
        weights = jax.lax.exp(lp - jax.scipy.special.logsumexp(lp))
        c += cdf(x, qs, weights) * 1/N
        c.block_until_ready()
        t1 = time()
        print(f"Finished {i} in {t1-t0:.3f}s")
    return c

# c = cdf(x, qs, weights)
# c.block_until_ready()
c = cdf_cruncher(qs, 10)
# plt.plot(qs, c)
# plt.show()

x = x[weights > 1e-9]
weights = weights[weights > 1e-9]
print(x.shape)

kde = jax.scipy.stats.gaussian_kde(x, weights=weights)
ps = kde(qs)
plt.plot(qs, ps, color="tab:blue")
plt.plot(qs, jnp.hstack([jnp.array(0.),jnp.diff(c)]) / (qs[1] - qs[0]), color="tab:orange")
plt.show()