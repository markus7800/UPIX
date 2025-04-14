from dccxjax.core import SLP
from ..types import PRNGKey, Trace, Traces, StackedTrace, StackedTraces
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import dccxjax.distributions as dist
from typing import Optional, Set

__all__ = [
    "estimate_Z_for_SLP_from_prior",
    "estimate_Z_for_SLP_from_mcmc",
    "estimate_Z_for_SLP_from_gaussian_mixture",
]

def estimate_Z_for_SLP_from_prior(slp: SLP, N: int, rng_key: PRNGKey):
    rng_keys = jax.random.split(rng_key, N)
    log_weights, in_support = jax.vmap(slp._gen_likelihood_weight)(rng_keys)

    # weights = jnp.exp(log_weights)
    # weights_sum = jnp.sum(weights)
    # ess = (weights_sum ** 2) / jnp.sum(weights ** 2)

    log_weights_sum = jax.scipy.special.logsumexp(log_weights)
    log_weights_squared_sum = jax.scipy.special.logsumexp(log_weights * 2)
    log_ess = log_weights_sum * 2 - log_weights_squared_sum
    # print(f"{ess=} {jnp.exp(log_ess)=}")

    frac_out_of_support = 1-jnp.mean(in_support)
    return jnp.exp(log_weights_sum) / N, jnp.exp(log_ess), frac_out_of_support

def estimate_Z_for_SLP_from_mcmc(
    slp: SLP, scale: float, samples_per_trace: int, seed: PRNGKey, *,
    Xs_unconstrained: Optional[Trace] = None, Xs_constrained: Optional[Trace] = None):
    assert (Xs_unconstrained is not None) ^ (Xs_constrained is not None)

    if Xs_unconstrained is not None:
        return estimate_Z_for_SLP_from_gaussian_mixture(slp, scale, samples_per_trace, seed, Xs_unconstrained, True)
    elif Xs_constrained is not None:
        return estimate_Z_for_SLP_from_gaussian_mixture(slp, scale, samples_per_trace, seed, Xs_constrained, False)
    else:
        raise Exception

def estimate_Z_for_SLP_from_gaussian_mixture(slp: SLP, scale: float, samples_per_point: int, seed: PRNGKey, centers: Trace, unconstrained: bool = True):
    assert slp.all_continuous()

    #@jax.jit
    def _log_IS_weight(rng_key: PRNGKey, X: Trace):
        X_flat, unravel_fn = ravel_pytree(X)
        Q = dist.Normal(X_flat, scale) # type: ignore
        X_prime_flat = Q.sample(rng_key)
        X_prime = unravel_fn(X_prime_flat)
        if unconstrained:
            return slp._unconstrained_log_prob(X_prime)[0] - Q.log_prob(X_prime_flat).sum()
        else:
            return slp.log_prob(X_prime) - Q.log_prob(X_prime_flat).sum()
    
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
    


def estimate_Z_for_SLP_from_sparse_mixture(slp: SLP, scale_cont: float, scale_disc: float, p_cont: float, p_disc: float, samples_per_point: int, seed: PRNGKey, centers: Trace, unconstrained: bool):
    all_continuous = slp.all_continuous()
    all_discrete = slp.all_discrete()

    _, some_entry = next(iter(centers.items()))
    N = some_entry.shape[0]

    @jax.jit
    def _log_IS_weight(rng_key: PRNGKey, X: Trace):
        sample_key, mask_key = jax.random.split(rng_key)

        if all_continuous:
            X_flat, unravel_fn = ravel_pytree(X)

            Z = dist.Normal(0, scale_cont).sample(sample_key, X_flat.shape)
            mask = dist.BernoulliProbs(p_cont).sample(mask_key, X_flat.shape)
            Q = dist.Normal(0, scale_cont).log_prob(Z).sum() + dist.BernoulliProbs(p_cont).log_prob(mask).sum()
            X_prime_flat = X_flat + mask * Z
            X_prime = unravel_fn(X_prime_flat)
            if unconstrained:
                return slp._unconstrained_log_prob(X_prime)[0] - Q
            else:
                return slp.log_prob(X_prime) - Q
            
        else:
            
            is_discrete = slp.get_is_discrete_map()

            X_discrete = {addr: val for addr, val in X.items() if is_discrete[addr]}
            X_flat_discrete, discrete_unravel_fn = ravel_pytree(X_discrete)
            B = 2 * dist.BernoulliProbs(0.5).sample(sample_key, X_flat_discrete.shape) - 1
            mask_discrete = dist.BernoulliProbs(p_disc).sample(mask_key, X_flat_discrete.shape)
            new_X_flat_discrete = X_flat_discrete + mask_discrete * B

            X_continuous = {addr: val for addr, val in X.items() if not is_discrete[addr]}
            X_flat_continuous, continuous_unravel_fn = ravel_pytree(X_continuous)
            Z = dist.Normal(0, scale_cont).sample(sample_key, X_flat_continuous.shape)
            mask_continuous = dist.BernoulliProbs(p_cont).sample(mask_key, X_flat_continuous.shape)
            new_X_flat_continuous = X_flat_continuous + mask_continuous * Z

            X_prime = discrete_unravel_fn(new_X_flat_discrete) | continuous_unravel_fn(new_X_flat_continuous)

            # Q = dist.Normal(0, scale_cont).log_prob(Z).sum()# + dist.BernoulliProbs(p_cont).log_prob(mask_continuous).sum()# + dist.BernoulliProbs(0.5).log_prob(B).sum() + dist.BernoulliProbs(p_disc).log_prob(mask_discrete).sum() + 
            # jax.debug.print("{x}", x= dist.BernoulliProbs(p_disc).log_prob(mask_discrete).sum())

            def _Q(center: Trace):
                center_continuous = {addr: val for addr, val in center.items() if not is_discrete[addr]}
                center_flat, _ = ravel_pytree(center_continuous)
                return dist.Normal(center_flat, scale_cont).log_prob(new_X_flat_continuous).sum()
            Qs = jax.lax.map(_Q, centers)
            Q = jax.scipy.special.logsumexp(Qs) - jax.lax.log(float(N))

            P = slp._unconstrained_log_prob(X_prime)[0] if unconstrained else slp.log_prob(X_prime)

            # jax.debug.print("P = {P} Q = {Q} \n\tX={X} \n\tX_prime = {X_prime}", P=P, Q=Q, X=X, X_prime = X_prime)
            # jax.debug.print("P = {P} Q = {Q} \n\tX_continuous={X_continuous}", P=P, Q=Q, X_continuous=X_continuous)

            return P - Q
        

    @jax.jit
    def _weight_sum_for_Xs(rng_key: PRNGKey):
        rng_keys = jax.random.split(rng_key, N)
        log_weights = jax.lax.map(lambda x: _log_IS_weight(*x), ((rng_keys, centers)))
        log_weights_sum = jax.scipy.special.logsumexp(log_weights)
        log_weights_squared_sum = jax.scipy.special.logsumexp(log_weights * 2)
        return log_weights_sum, log_weights_squared_sum, jnp.sum(jnp.isinf(log_weights))
                       
    log_weights_sums, log_weights_squared_sums, out_of_supports = jax.vmap(_weight_sum_for_Xs)(jax.random.split(seed, samples_per_point))

    log_weights_sum = jax.scipy.special.logsumexp(log_weights_sums)
    log_weights_squared_sum = jax.scipy.special.logsumexp(log_weights_squared_sums)
    frac_out_of_support = jnp.sum(out_of_supports) / (N * samples_per_point)

    log_ess = log_weights_sum * 2 - log_weights_squared_sum
    print(f"{log_weights_sum=}")

    return jnp.exp(log_weights_sum) / (N * samples_per_point), jnp.exp(log_ess), frac_out_of_support


def estimate_Z_for_SLP_from_mcmc_2(
    slp: SLP, scale: float, samples_per_trace: int, seed: PRNGKey, *,
    Xs_unconstrained: Optional[StackedTraces] = None, Xs_constrained: Optional[StackedTraces] = None,
    addresses_to_keep_constant: Set[str] = set()
):
    assert (Xs_unconstrained is not None) ^ (Xs_constrained is not None)

    if Xs_unconstrained is not None:
        return estimate_Z_for_SLP_from_unconstrained_gaussian_mixture_2(slp, scale, samples_per_trace, seed, Xs_unconstrained, True)
    elif Xs_constrained is not None:
        return estimate_Z_for_SLP_from_unconstrained_gaussian_mixture_2(slp, scale, samples_per_trace, seed, Xs_constrained, False)
    else:
        raise Exception

def _log_IS_weight_gaussian_mixture(slp: SLP, seed: PRNGKey, centers: Traces, scale: float, unconstrained: bool):
    centers_flat, unravel_centers_fn = ravel_pytree(centers.data)
    centers_flat = centers_flat.reshape(centers_flat.size // centers.N, centers.N)
    # centers_flat[:,j] corresponds to center j

    def _log_IS_weight(rng_key: PRNGKey, center: Trace):
        # print(rng_key)
        # print(center)
        X_flat, unravel_fn = ravel_pytree(center)
        Q = dist.Normal(X_flat, scale) # type: ignore
        X_prime_flat = Q.sample(rng_key) 
        X_prime = unravel_fn(X_prime_flat)

        # print(f"{X_prime_flat.shape=}")

        # score X_prime_flat with respect to all centers
        qs = dist.Normal(centers_flat, scale).log_prob(X_prime_flat.reshape(X_prime_flat.shape + (1,)))  # type: ignore
        q = jax.scipy.special.logsumexp(qs)

        if unconstrained:
            return slp._unconstrained_log_prob(X_prime)[0] - q
        else:
            return slp.log_prob(X_prime) - q

    return jax.vmap(_log_IS_weight)(jax.random.split(seed, centers.N), centers.data)

def estimate_Z_for_SLP_from_unconstrained_gaussian_mixture_2(slp: SLP, scale: float, samples_per_point: int, seed: PRNGKey, centers: StackedTraces, unconstrained: bool):
    assert slp.all_continuous()


    # log_weights = jax.vmap(_log_IS_weight_gaussian_mixture, (None, 0, 1, None, None, None))(slp, seed, centers, scale, unconstrained)

    #@jax.jit
    # def _weight_sum_for_Xs(rng_key: PRNGKey):
    #     rng_keys = jax.random.split(rng_key, N)
    #     log_weights = jax.vmap(_log_IS_weight)(rng_keys, centers)
    #     weights = jnp.exp(log_weights)
    #     return jnp.sum(weights), jnp.sum(weights ** 2), jnp.sum(jnp.isinf(log_weights))
                       
    # weights_sums, weights_squared_sums, out_of_supports = jax.vmap(_weight_sum_for_Xs)(jax.random.split(seed, samples_per_point))

    # weights_sum = jnp.sum(weights_sums)
    # weights_squared_sum = jnp.sum(weights_squared_sums)
    # frac_out_of_support = jnp.sum(out_of_supports) / (N * samples_per_point)

    # ess = (weights_sum ** 2) / weights_squared_sum
    # return weights_sum / (N * samples_per_point), ess, frac_out_of_support
    

