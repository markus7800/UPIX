from dccxjax.core import SLP
from dccxjax.types import PRNGKey, Trace, Traces, StackedTrace, StackedTraces
import jax
import jax.numpy as jnp
from dccxjax.parallelisation import ParallelisationConfig, VectorisationType, vectorise, parallel_run, SHARDING_AXIS
from dccxjax.jax_utils import smap_vmap

__all__ = [
    "estimate_log_Z_for_SLP_from_prior",
]


def estimate_log_Z_for_SLP_from_prior(slp: SLP, N: int, rng_key: PRNGKey, pconfig: ParallelisationConfig):
    rng_keys = jax.random.split(rng_key, N)
    
    # this does not really make a difference
    if pconfig.vectorisation == VectorisationType.LocalVMAP:
        _gen_likelihood_weight = jax.vmap(slp._gen_likelihood_weight)
    elif pconfig.vectorisation == VectorisationType.LocalSMAP:
        _gen_likelihood_weight = smap_vmap(slp._gen_likelihood_weight, axis_name=SHARDING_AXIS)
    else:
        _gen_likelihood_weight = slp._gen_likelihood_weight
    vectorised_fn = vectorise(_gen_likelihood_weight, 0, 0, N, pconfig.vectorisation, batch_size=pconfig.batch_size)    
    log_weights, in_support = parallel_run(vectorised_fn, (rng_keys,), N, pconfig.vectorisation)
    
    # log_weights, in_support = jax.vmap(slp._gen_likelihood_weight)(rng_keys)

    log_weights_sum = jax.scipy.special.logsumexp(log_weights)
    log_weights_squared_sum = jax.scipy.special.logsumexp(log_weights * 2)
    log_ess = log_weights_sum * 2 - log_weights_squared_sum

    frac_in_support = in_support.sum() / N
    return log_weights_sum - jax.lax.log(float(N)), jnp.exp(log_ess), frac_in_support


