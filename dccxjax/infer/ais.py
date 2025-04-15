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

    

def run_ais(slp: SLP, ais: AISConfig, seed: PRNGKey, xs: Trace, lp: jax.Array, N: int):
    # during inference we have to hold N traces in memory, so there is no benefit of run_ais above (if it would work)

    assert ais.tempering_schedule[0] == 0.  # log prior
    assert ais.tempering_schedule[-1] == 1. # log joint

    prior_key, tempering_key = jax.random.split(seed)
    prior_kernel_init = MCMCState(jnp.array(0,int), jnp.array(0.,float), xs, lp, broadcast_jaxtree([],(N,)))
    prior_keys = jax.random.split(prior_key, ais.n_prior_mcmc_steps)
    last_prior_state, _ = jax.lax.scan(ais.prior_kernel, prior_kernel_init, prior_keys)


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

        next_inference_carry, _ = ais.tempering_kernel(current_inference_carry, tempering_key)
        tempered_log_prob_after_step = next_inference_carry.log_prob

        next_log_weight = current_log_weight + tempered_log_prob_before_step - tempered_log_prob_after_step
    
        return (next_inference_carry, next_log_weight), None


    tempering_keys = jax.random.split(tempering_key, ais.tempering_schedule.size)

    (last_tempering_state, log_weight), _ = jax.lax.scan(tempering_step, (last_prior_state, broadcast_jaxtree(jnp.array(0., float), (N,))), (tempering_keys, ais.tempering_schedule))

    log_prior = last_prior_state.log_prob
    log_joint = last_tempering_state.log_prob

    return log_weight + log_joint - log_prior


    