from dccxjax.core import SLP
from ..types import PRNGKey, Trace
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpyro.distributions as dist

__all__ = [
    "estimate_Z_for_SLP_from_prior",
    "estimate_Z_for_SLP_from_mcmc",
]

def estimate_Z_for_SLP_from_prior(slp: SLP, N: int, rng_key: PRNGKey):
    rng_keys = jax.random.split(rng_key, N)
    log_weights = jax.vmap(slp._gen_likelihood_weight)(rng_keys)
    weights = jnp.exp(log_weights)
    weights_sum = jnp.sum(weights)
    ess = (weights_sum ** 2) / jnp.sum(weights ** 2)
    frac_out_of_support = jnp.mean(jnp.isinf(log_weights))
    return weights_sum / N, ess, frac_out_of_support

def estimate_Z_for_SLP_from_mcmc(slp: SLP, scale: float, samples_per_trace: int, seed: PRNGKey, Xs: Trace):
    @jax.jit
    def _log_IS_weight(rng_key: PRNGKey, X: Trace):
        X_flat, unravel_fn = ravel_pytree(X)
        Q = dist.Normal(X_flat, scale) # type: ignore
        X_prime_flat = Q.sample(rng_key)
        X_prime = unravel_fn(X_prime_flat)
        return slp._log_prob(X_prime) - Q.log_prob(X_prime_flat).sum()
    
    _, some_entry = next(iter(Xs.items()))
    N = some_entry.shape[0]
    @jax.jit
    def _weight_sum_for_Xs(rng_key: PRNGKey):
        rng_keys = jax.random.split(rng_key, N)
        log_weights = jax.vmap(_log_IS_weight)(rng_keys, Xs)
        weights = jnp.exp(log_weights)
        return jnp.sum(weights), jnp.sum(weights ** 2), jnp.sum(jnp.isinf(log_weights))
                       
    weights_sums, weights_squared_sums, out_of_supports = jax.vmap(_weight_sum_for_Xs)(jax.random.split(seed, samples_per_trace))

    weights_sum = jnp.sum(weights_sums)
    weights_squared_sum = jnp.sum(weights_squared_sums)
    frac_out_of_support = jnp.sum(out_of_supports) / (N * samples_per_trace)

    ess = (weights_sum ** 2) / weights_squared_sum
    return weights_sum / (N * samples_per_trace), ess, frac_out_of_support
    