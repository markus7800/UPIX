#%%
from dccxjax import *
import numpyro.distributions as dist
import jax.numpy as jnp

import logging
# setup_logging(logging.DEBUG)

@model
def mm(p):
    u = sample("u", dist.Uniform())
    # print(f"{u=}")
    if u < p:
        m = sample("m1", dist.Normal(-1.,1.))
    else:
        m = sample("m2", dist.Normal(2.,0.5))
    sample("y", dist.Normal(m, 0.5), observed=0.5)


p = 0.7

m: Model = mm(p) # type: ignore


U = jnp.linspace(0.,1.,1000)
M = jnp.linspace(-6.,6.,5000)
Us, Ms = jnp.meshgrid(U, M)

prior1 = jnp.exp(dist.Uniform().log_prob(Us) + dist.Normal(-1.,1.).log_prob(Ms))
prior2 = jnp.exp(dist.Uniform().log_prob(Us) + dist.Normal(2.,0.5).log_prob(Ms))
prior = prior2.at[Us < p].set(prior1[Us < p])
print(jnp.trapezoid(jnp.trapezoid(prior, M, axis=0), U, axis=0))

likeli = jnp.exp(dist.Normal(Ms, 0.5).log_prob(0.5))

Z1 = jnp.trapezoid(jnp.trapezoid(prior * likeli.at[Us >= p].set(0.), M, axis=0), U, axis=0)
Z2 = jnp.trapezoid(jnp.trapezoid(prior * likeli.at[Us < p].set(0.), M, axis=0), U, axis=0)
Z = jnp.trapezoid(jnp.trapezoid(prior * likeli, M, axis=0), U, axis=0)
print(f"{Z=}")
print(f"{Z1=}")
print(f"{Z2=}")
print(f"{Z1+Z2=}")


def estimate_evidence(model: Model, rng_keys: PRNGKey):
    lps = []  
    for rng_key in rng_keys:
        with GenerateCtx(rng_key) as ctx:
            model()
            lps.append(ctx.log_likelihood)
    return jnp.mean(jnp.exp(jnp.array(lps)))

keys = jax.random.split(jax.random.PRNGKey(0), 1_000_000)
# Z_est = estimate_evidence(m, keys[:10_000])
# print(f"{Z_est=}")

active_slps: List[SLP] = []
for key in range(10):
    rng_key = jax.random.PRNGKey(key)
    X = sample_from_prior(m, rng_key)
    slp = slp_from_X(m, X)

    if all(slp.path_indicator(X) == 0 for slp in active_slps):
        active_slps.append(slp)
#%%
print()
print()
print("active_slps: ")
for slp in active_slps:
    print(slp)
    Z_slp_est = estimate_Z_for_SLP(slp, keys)
    print(f"{Z_slp_est=}")
    result = mcmc(slp, InferenceStep(AllVariables(), RandomWalk(gaussian_random_walk(0.1))), 10, 2, jax.random.PRNGKey(0))
    print(result)


