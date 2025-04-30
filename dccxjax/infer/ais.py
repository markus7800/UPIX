import jax
from typing import NamedTuple, Tuple
from ..types import PRNGKey, Trace, FloatArray
from ..core.model_slp import SLP
from .mcmc import MCMCKernel, MCMCState
from ..utils import broadcast_jaxtree
import jax.numpy as jnp

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
        prior_kernel_init = MCMCState(jnp.array(0,int), jnp.array(0.,float), xs, lp, broadcast_jaxtree([],(N,)))
        prior_keys = jax.random.split(prior_key, config.n_prior_mcmc_steps)
        last_prior_state, _ = jax.lax.scan(config.prior_kernel, prior_kernel_init, prior_keys)
    else:
        last_prior_state = MCMCState(jnp.array(0,int), jnp.array(0.,float), xs, lp, broadcast_jaxtree([],(N,)))

    def tempering_step(carry: Tuple[MCMCState,FloatArray], tempering: Tuple[PRNGKey,FloatArray]) -> Tuple[Tuple[MCMCState,FloatArray], None]:
        inference_carry_from_prev_kernel, current_log_weight = carry
        tempering_key, temperature = tempering

        # log weights with respect to current tempering
        log_prior_before, log_likelihood_before, _ = jax.vmap(slp._log_prior_likeli_pathcond)(inference_carry_from_prev_kernel.position)
        tempered_log_prob_before_step = log_prior_before + temperature * log_likelihood_before

        current_inference_carry = MCMCState(
            inference_carry_from_prev_kernel.iteration,
            temperature,
            inference_carry_from_prev_kernel.position,
            tempered_log_prob_before_step,
            broadcast_jaxtree([],(N,))
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

def run_smc(slp: SLP, config: SMCConfig, seed: PRNGKey, xs: Trace, lp: jax.Array, N: int):
    # during inference we have to hold N traces in memory, so there is no benefit of run_ais above (if it would work)

    assert config.tempering_schedule[0] == 0.  # log prior
    assert config.tempering_schedule[-1] == 1. # log joint
    
    prior_key, tempering_key = jax.random.split(seed)

    init_state = SMCCarry(
        MCMCState(jnp.array(0,int), jnp.array(0.,float), xs, lp, broadcast_jaxtree([],(N,))),
        jnp.zeros((N,), float),
    )

    def smc_tempering_step(carry: SMCCarry, tempering: Tuple[PRNGKey,FloatArray]) -> Tuple[SMCCarry, FloatArray]:
        inference_state_from_prev_kernel = carry.mcmc_state
        current_log_particle_weight = carry.log_particle_weight
        tempering_key, temperature = tempering

        # log weights with respect to current tempering
        log_prior_before, log_likelihood_before, _ = jax.vmap(slp._log_prior_likeli_pathcond)(inference_state_from_prev_kernel.position)
        tempered_log_prob_current_position = log_prior_before + temperature * log_likelihood_before

        log_w_hat = tempered_log_prob_current_position - inference_state_from_prev_kernel.log_prob # p_k(phi_{k-1}) / p_{k-1}(phi_{k-1})
        log_particle_weight = current_log_particle_weight + log_w_hat
        log_particle_weight_sum = jax.scipy.special.logsumexp(log_particle_weight)
        log_ess = log_particle_weight_sum * 2 - jax.scipy.special.logsumexp(log_particle_weight*2)

        current_inference_state = MCMCState(
            inference_state_from_prev_kernel.iteration,
            temperature,
            inference_state_from_prev_kernel.position,
            tempered_log_prob_current_position,
            broadcast_jaxtree([],(N,))
        )

        next_inference_state, _ = config.tempering_kernel(current_inference_state, tempering_key)

    
        return SMCCarry(next_inference_state, log_particle_weight), log_ess


    tempering_keys = jax.random.split(tempering_key, config.tempering_schedule.size)

    last_tempering_state, log_ess = jax.lax.scan(smc_tempering_step, init_state, (tempering_keys, config.tempering_schedule))


    return last_tempering_state.log_particle_weight


    