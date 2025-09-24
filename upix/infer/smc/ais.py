import jax
from typing import NamedTuple, Tuple
from upix.types import PRNGKey, Trace, FloatArray, IntArray
from upix.core.model_slp import SLP
from upix.infer.mcmc.mcmc_core import MCMCKernel, MCMCState, CarryStats
from upix.utils import broadcast_jaxtree
import jax.numpy as jnp

__all__ = [
]

class AISConfig(NamedTuple):
    prior_kernel: MCMCKernel
    n_prior_mcmc_steps: int
    tempering_kernel: MCMCKernel
    tempering_schedule: jax.Array

def run_ais(slp: SLP, config: AISConfig, seed: PRNGKey, xs: Trace, lp: jax.Array, N: int):
    # during inference we have to hold N traces in memory, so there is no benefit of run_ais above (if it would work)

    assert config.tempering_schedule[0] == 0.  # log prior
    assert config.tempering_schedule[-1] == 1. # log joint

    prior_key, tempering_key = jax.random.split(seed)
    
    if config.n_prior_mcmc_steps > 0:
        prior_kernel_init = MCMCState(jnp.array(0,int), jnp.array(0.,float), dict(), xs, lp, CarryStats(), None)
        prior_keys = jax.random.split(prior_key, config.n_prior_mcmc_steps)
        last_prior_state, _ = jax.lax.scan(config.prior_kernel, prior_kernel_init, prior_keys)
    else:
        last_prior_state = MCMCState(jnp.array(0,int), jnp.array(0.,float), dict(), xs, lp, CarryStats(), None)

    def tempering_step(carry: Tuple[MCMCState,FloatArray], tempering: Tuple[PRNGKey,FloatArray]) -> Tuple[Tuple[MCMCState,FloatArray], None]:
        inference_carry_from_prev_kernel, current_log_weight = carry
        tempering_key, temperature = tempering

        # log weights with respect to current tempering
        log_prior_before, log_likelihood_before, _ = jax.vmap(slp._log_prior_likeli_pathcond)(inference_carry_from_prev_kernel.position, dict())
        tempered_log_prob_before_step = log_prior_before + temperature * log_likelihood_before

        current_inference_carry = MCMCState(
            inference_carry_from_prev_kernel.iteration,
            temperature,
            dict(),
            inference_carry_from_prev_kernel.position,
            tempered_log_prob_before_step,
            CarryStats(),
            None
        )

        next_inference_carry, _ = config.tempering_kernel(current_inference_carry, tempering_key)
        tempered_log_prob_after_step = next_inference_carry.log_prob

        next_log_weight = current_log_weight + tempered_log_prob_before_step - tempered_log_prob_after_step
    
        return (next_inference_carry, next_log_weight), None


    tempering_keys = jax.random.split(tempering_key, config.tempering_schedule.size)

    (last_tempering_state, log_weight), _ = jax.lax.scan(tempering_step, (last_prior_state, broadcast_jaxtree(jnp.array(0., float), (N,))), (tempering_keys, config.tempering_schedule))

    log_prior = last_prior_state.log_prob
    log_joint = last_tempering_state.log_prob

    return log_weight + log_joint - log_prior, last_tempering_state.position


    
class SMCConfig(NamedTuple):
    tempering_kernel: MCMCKernel
    tempering_schedule: jax.Array

class SMCCarry(NamedTuple):
    mcmc_state: MCMCState
    log_particle_weight: FloatArray



# from blackjax/smc/resampling.py
def sorted_uniforms(rng_key: PRNGKey, n) -> FloatArray:
    # Credit goes to Nicolas Chopin
    es = jax.random.exponential(rng_key, (n + 1,))
    z = jnp.cumsum(es)
    return z[:-1] / z[-1]

def multinomial(rng_key: PRNGKey, weights: FloatArray, num_samples: int) -> IntArray:
    n = weights.shape[0]
    linspace = sorted_uniforms(rng_key, num_samples)
    cumsum = jnp.cumsum(weights)
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)

def systematic_or_stratified(rng_key: PRNGKey, weights: FloatArray, num_samples: int, is_systematic: bool) -> IntArray:
    n = weights.shape[0]
    if is_systematic:
        u = jax.random.uniform(rng_key, ())
    else:
        u = jax.random.uniform(rng_key, (num_samples,))
    cumsum = jnp.cumsum(weights)
    linspace = (jnp.arange(num_samples, dtype=weights.dtype) + u) / num_samples
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)

def run_smc(slp: SLP, config: SMCConfig, seed: PRNGKey, xs: Trace, lp: jax.Array, N: int, resampling: str = "adaptive"):
    # during inference we have to hold N traces in memory, so there is no benefit of run_ais above (if it would work)
    assert resampling in ("adaptive", "always", "never")

    # assert config.tempering_schedule[0] == 0.  # log prior
    assert config.tempering_schedule[-1] == 1. # log joint
    
    # prior_key, tempering_key = jax.random.split(seed)
    tempering_key = seed

    init_state = SMCCarry(
        MCMCState(jnp.array(0,int), jnp.array(0.,float), dict(), xs, lp, CarryStats(), None),
        jnp.zeros((N,), float),
    )

    def smc_tempering_step(carry: SMCCarry, tempering: Tuple[PRNGKey,FloatArray]):# -> Tuple[SMCCarry, FloatArray]:
        # following llorente 2023

        inference_state_from_prev_kernel = carry.mcmc_state
        current_log_particle_weight = carry.log_particle_weight
        key, temperature = tempering
        tempering_key, resample_key = jax.random.split(key)

        # log weights with respect to current tempering
        log_prior_before, log_likelihood_before, _ = jax.vmap(slp._log_prior_likeli_pathcond)(inference_state_from_prev_kernel.position, dict())
        tempered_log_prob_current_position = log_prior_before + temperature * log_likelihood_before # p_k(phi_{k-1})


        current_inference_state = MCMCState(
            inference_state_from_prev_kernel.iteration,
            temperature,
            dict(),
            inference_state_from_prev_kernel.position,
            tempered_log_prob_current_position,
            CarryStats(),
            None
        )

        next_inference_state, _ = config.tempering_kernel(current_inference_state, tempering_key) # leaves p_k invariant

        log_w_hat = tempered_log_prob_current_position - inference_state_from_prev_kernel.log_prob # p_k(phi_{k-1}) / p_{k-1}(phi_{k-1})
        # log_Z = jax.scipy.special.logsumexp(log_w_hat + (current_log_particle_weight - jax.scipy.special.logsumexp(current_log_particle_weight)))

        # jax.debug.print("{ls} tempered_log_prob_current_position={a} inference_state_from_prev_kernel.log_prob={b} log_w_hat={c}, x={x} lp={d}", ls = jax.scipy.special.logsumexp(carry.log_particle_weight), a=tempered_log_prob_current_position[:5], b=inference_state_from_prev_kernel.log_prob[:5], c=log_w_hat[:5], x=next_inference_state.position["x"][:5], d=next_inference_state.log_prob[:5])
        # jax.debug.print("x={x} lp={a}", x=next_inference_state.position["x"][:5], a=next_inference_state.log_prob[:5])

        log_particle_weight = current_log_particle_weight + log_w_hat
        log_particle_weight_sum = jax.scipy.special.logsumexp(log_particle_weight)
        log_ess = log_particle_weight_sum * 2 - jax.scipy.special.logsumexp(log_particle_weight*2)

        # high (quadratic?) memory consumption
        # resample_ixs = jax.random.categorical(resample_key, log_particle_weight, shape=(N,)) # uses gumbel reparametrisation trick

        if resampling != "never":
            # jax.debug.print("log_ess={log_ess}", log_ess=log_ess)

            def do_resample(next_inference_state: MCMCState, log_particle_weight: FloatArray):
                resample_ixs = multinomial(resample_key, jax.lax.exp(log_particle_weight - log_particle_weight_sum), N)
                # resample_ixs = systematic_or_stratified(resample_key, jax.lax.exp(log_particle_weight - log_particle_weight_sum), N, False)
            
                next_inference_state = MCMCState(next_inference_state.iteration, next_inference_state.temperature, dict(),
                    jax.tree.map(lambda v: v[resample_ixs,...], next_inference_state.position),
                    next_inference_state.log_prob[resample_ixs],
                    CarryStats(),
                    next_inference_state.infos
                )
                log_particle_weight = jax.lax.broadcast(log_particle_weight_sum - jax.lax.log(float(N)), (N,)) # proper weighting
                # log_particle_weight = jnp.zeros((N,),float) # improper weighting
                return next_inference_state, log_particle_weight
            
            def no_resample(next_inference_state: MCMCState, log_particle_weight: FloatArray):
                return next_inference_state, log_particle_weight
            
            if resampling == "always":
                next_inference_state, log_particle_weight = do_resample(next_inference_state, log_particle_weight)
            else:
                next_inference_state, log_particle_weight = jax.lax.cond(log_ess < jax.lax.log(N / 2.0), do_resample, no_resample, next_inference_state, log_particle_weight)

        return SMCCarry(next_inference_state, log_particle_weight), log_ess


    tempering_keys = jax.random.split(tempering_key, config.tempering_schedule.size)
    
    last_tempering_state, log_ess = jax.lax.scan(smc_tempering_step, init_state, (tempering_keys, config.tempering_schedule))
    # log_Z = log_Zs.sum()
    # print(f"{log_Z=} {jax.lax.exp(log_Z)=}")

    return last_tempering_state.log_particle_weight, last_tempering_state.mcmc_state.position, log_ess


    