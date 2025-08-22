#%%
import jax
import jax.numpy as jnp
from typing import NamedTuple
from time import time
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

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
def pedestrian(rng_key: jax.Array):
    rng_key, sample_key = jax.random.split(rng_key)
    start = jax.random.uniform(sample_key) * 3.

    end_state = jax.lax.while_loop(cond, body_fun, PedWhileState(rng_key, start, jnp.array(0.), jnp.array(0)))

    likelihood = jax.scipy.stats.norm.logpdf(1.1, end_state.distance, 0.1)

    return {"start": start, "lp": likelihood}

def pedestrian_batched(seed: jax.Array, N: int, M: int):
    rng_keys = jax.random.split(seed, (N, M))
    return jax.vmap(jax.vmap(pedestrian))(rng_keys)

# pedestrian_batched(jax.random.key(0), N, 1) == [pedestrian1(key) for key in jax.random.split(jax.random.key(0), N)]

#%%
t0 = time()
# result = pedestrian_batched(jax.random.key(0), 1_000_000, 1) # not batching while loop is faster on CPU
result = jax.vmap(pedestrian)(jax.random.split(jax.random.key(0), 10_000_000))
result["start"].block_until_ready()
x = result["start"].reshape(-1)
lp = result["lp"].reshape(-1)
t1 = time()
print(f"Finished in {t1-t0:.3f}s")


weights = jax.lax.exp(lp - jax.scipy.special.logsumexp(lp))
print(weights.sum())
plt.hist(x, weights=weights, density=True, bins=100)
plt.grid(True)
plt.yticks(jnp.arange(0.,1.2,0.05))
plt.show()

#%%


@jax.jit
def cdf(x, qs, weights: jax.Array):
    def _cdf(q):
        return jnp.where(x < q, weights, jax.numpy.zeros_like(weights)).sum()
    return jax.lax.map(_cdf, qs)

def cdf_cruncher(qs, N, M):
    rng_keys = jax.random.split(jax.random.key(0), N)
    c = jnp.zeros_like(qs)
    for (i,key) in tqdm(enumerate(rng_keys),total=N):
        t0 = time()
        result = pedestrian_batched(key, M, 1)
        x = result["start"].reshape(-1)
        lp = result["lp"].reshape(-1)
        weights = jax.lax.exp(lp - jax.scipy.special.logsumexp(lp))
        c += cdf(x, qs, weights) * 1/N
        t1 = time()
        # print(f"Finished {i} in {t1-t0:.3f}s")
    return c

start_linspace = jnp.linspace(0., 3., 100)

gt_cdf = cdf_cruncher(start_linspace, 1000, 10_000_000)
plt.plot(start_linspace, gt_cdf)
plt.show()

x = x[weights > 1e-9]
weights = weights[weights > 1e-9]
print(x.shape)

kde = jax.scipy.stats.gaussian_kde(x, weights=weights)
kde_pdf = kde(start_linspace)
plt.plot(start_linspace, kde_pdf, color="tab:blue")
gt_pdf = jnp.hstack([jnp.array(0.),jnp.diff(gt_cdf)]) / (start_linspace[1] - start_linspace[0])
plt.plot(start_linspace, gt_pdf, color="tab:orange")
plt.show()

jnp.save("evaluation/pedestrian/gt_xs.npy", start_linspace)
jnp.save("evaluation/pedestrian/gt_pdf.npy", gt_pdf)
jnp.save("evaluation/pedestrian/gt_cdf.npy", gt_cdf)