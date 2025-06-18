from dccxjax.core import SLP
from dccxjax.types import PRNGKey, Trace, Traces, StackedTrace, StackedTraces
import jax
import jax.numpy as jnp

__all__ = [
    "estimate_Z_for_SLP_from_prior",
    "estimate_log_Z_for_SLP_from_prior",
]

def estimate_log_Z_for_SLP_from_prior(slp: SLP, N: int, rng_key: PRNGKey):
    rng_keys = jax.random.split(rng_key, N)
    log_weights, in_support = jax.vmap(slp._gen_likelihood_weight)(rng_keys)

    # weights = jnp.exp(log_weights)
    # weights_sum = jnp.sum(weights)
    # ess = (weights_sum ** 2) / jnp.sum(weights ** 2)

    log_weights_sum = jax.scipy.special.logsumexp(log_weights)
    log_weights_squared_sum = jax.scipy.special.logsumexp(log_weights * 2)
    log_ess = log_weights_sum * 2 - log_weights_squared_sum
    # print(f"{ess=} {jnp.exp(log_ess)=}")

    frac_in_support = in_support.sum() / N
    return log_weights_sum - jax.lax.log(float(N)), jnp.exp(log_ess), frac_in_support

def estimate_Z_for_SLP_from_prior(slp: SLP, N: int, rng_key: PRNGKey):
    log_Z, ESS, frac_in_support = estimate_log_Z_for_SLP_from_prior(slp, N, rng_key)
    return jax.lax.exp(log_Z), ESS, frac_in_support


