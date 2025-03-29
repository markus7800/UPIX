from dccxjax.core import SLP
from ..types import PRNGKey, Trace
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import dccxjax.distributions as dist
from typing import Optional

__all__ = [
    "estimate_Z_for_SLP_from_prior",
    "estimate_Z_for_SLP_from_mcmc",
    "estimate_Z_for_SLP_from_unconstrained_gaussian_mixture",
]

def estimate_Z_for_SLP_from_prior(slp: SLP, N: int, rng_key: PRNGKey):
    rng_keys = jax.random.split(rng_key, N)
    log_weights = jax.vmap(slp._gen_likelihood_weight)(rng_keys)

    # weights = jnp.exp(log_weights)
    # weights_sum = jnp.sum(weights)
    # ess = (weights_sum ** 2) / jnp.sum(weights ** 2)

    log_weights_sum = jax.scipy.special.logsumexp(log_weights)
    log_weights_squared_sum = jax.scipy.special.logsumexp(log_weights * 2)
    log_ess = log_weights_sum * 2 - log_weights_squared_sum
    # print(f"{ess=} {jnp.exp(log_ess)=}")

    frac_out_of_support = jnp.mean(jnp.isinf(log_weights))
    return jnp.exp(log_weights_sum) / N, jnp.exp(log_ess), frac_out_of_support

def estimate_Z_for_SLP_from_mcmc(
    slp: SLP, scale: float, samples_per_trace: int, seed: PRNGKey, *,
    Xs_unconstrained: Optional[Trace] = None, Xs_constrained: Optional[Trace] = None
):
    assert (Xs_unconstrained is not None) ^ (Xs_constrained is not None)
    if Xs_unconstrained is not None:
        return estimate_Z_for_SLP_from_unconstrained_gaussian_mixture(slp, scale, samples_per_trace, seed, Xs_unconstrained, True)
    elif Xs_constrained is not None:
        return estimate_Z_for_SLP_from_unconstrained_gaussian_mixture(slp, scale, samples_per_trace, seed, Xs_constrained, False)
    else:
        raise Exception


def estimate_Z_for_SLP_from_unconstrained_gaussian_mixture(slp: SLP, scale: float, samples_per_point: int, seed: PRNGKey, centers: Trace, unconstrained: bool = True):
    all_continuous = slp.all_continuous()
    is_discrete = {addr: jnp.array(val) for addr, val in slp.get_is_discrete_map().items()}

    #@jax.jit
    def _log_IS_weight(rng_key: PRNGKey, X: Trace):
        if all_continuous:
            X_flat, unravel_fn = ravel_pytree(X)
            Q = dist.Normal(X_flat, scale) # type: ignore
            X_prime_flat = Q.sample(rng_key)
            X_prime = unravel_fn(X_prime_flat)
            if unconstrained:
                return slp._unconstrained_log_prob(X_prime)[0] - Q.log_prob(X_prime_flat).sum()
            else:
                return slp.log_prob(X_prime) - Q.log_prob(X_prime_flat).sum()
        else:
            raise NotImplementedError
    
    _, some_entry = next(iter(centers.items()))
    N = some_entry.shape[0]
    #@jax.jit
    def _weight_sum_for_Xs(rng_key: PRNGKey):
        rng_keys = jax.random.split(rng_key, N)
        log_weights = jax.vmap(_log_IS_weight)(rng_keys, centers)
        weights = jnp.exp(log_weights)
        return jnp.sum(weights), jnp.sum(weights ** 2), jnp.sum(jnp.isinf(log_weights))
                       
    weights_sums, weights_squared_sums, out_of_supports = jax.vmap(_weight_sum_for_Xs)(jax.random.split(seed, samples_per_point))

    weights_sum = jnp.sum(weights_sums)
    weights_squared_sum = jnp.sum(weights_squared_sums)
    frac_out_of_support = jnp.sum(out_of_supports) / (N * samples_per_point)

    ess = (weights_sum ** 2) / weights_squared_sum
    return weights_sum / (N * samples_per_point), ess, frac_out_of_support
    