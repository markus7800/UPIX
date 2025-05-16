

import jax
import jax.numpy as jnp
from time import time
from evaluation.gp.kernels import *

xs = jnp.linspace(0,1,100)
ts = jax.random.normal(jax.random.PRNGKey(0), xs.shape)

def f(lengthscale, gamma, amplitude):
    dt = jax.lax.abs(ts.reshape(-1,1) - ts.reshape(1,-1))
    c = jax.lax.exp(- (dt/lengthscale)**gamma)
    cov_matrix = amplitude * c
    # cov_matrix = GammaExponential(lengthscale, gamma, amplitude).eval_cov_vec(xs)
    return jax.scipy.stats.multivariate_normal.logpdf(ts, jnp.zeros_like(xs), cov_matrix)

def fstar(t):
    return f(*t)

def step(rng_key, _):
    rng_key, k1, k2, k3 = jax.random.split(rng_key,4)

    lengthscale = jnp.exp(jax.random.normal(k1))
    gamma = jnp.exp(jax.random.normal(k2))
    amplitude = jnp.exp(jax.random.normal(k3))

    lp = f(lengthscale, gamma, amplitude)
    # g = jax.grad(fstar)((lengthscale, gamma, amplitude))

    return rng_key, lp

N_iter = 10_000

t0 = time()
r, _ = jax.lax.scan(step, jax.random.PRNGKey(0), length=N_iter)
r.block_until_ready()
t1 = time()
print(f"Finished in {t1-t0:.3f} s")


t0 = time()
r, _ = jax.lax.scan(jax.vmap(step), jax.lax.broadcast(jax.random.PRNGKey(0), (1,)), length=N_iter)
r.block_until_ready()
t1 = time()
print(f"Finished in {t1-t0:.3f} s")




# Ls = jax.vmap(f)(jax.lax.broadcast(jax.random.PRNGKey(0), (1,)))
# Ls.block_until_ready()
# t1 = time()
# print(f"Finished in {t1-t0:.3f} s")

# n = 100
# def step(rng_key, _):
#     rng_key, u_key = jax.random.split(rng_key)
#     U = jax.random.normal(u_key, (n,n))
#     A = 0.5 * (U + U.transpose())
#     L = jnp.linalg.cholesky(A)
#     return rng_key, L

# @jax.jit
# def f(seed):
#     return jax.lax.scan(step, seed, length=10_000)[1]

# t0 = time()
# Ls = f(jax.random.PRNGKey(0))
# Ls.block_until_ready()
# t1 = time()
# print(f"Finished in {t1-t0:.3f} s")

# t0 = time()
# Ls = jax.vmap(f)(jax.lax.broadcast(jax.random.PRNGKey(0), (1,)))
# Ls.block_until_ready()
# t1 = time()
# print(f"Finished in {t1-t0:.3f} s")
