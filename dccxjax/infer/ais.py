import jax
from typing import NamedTuple, Tuple
from ..types import PRNGKey, Trace
from ..core.model_slp import SLP
from .mcmc import MCMCKernel, InferenceCarry, InferenceState
from ..utils import broadcast_jaxtree
import jax.numpy as jnp

class AISConfig(NamedTuple):
    prior_kernel: MCMCKernel
    n_prior_mcmc_steps: int
    tempering_kernel: MCMCKernel
    tempering_schedule: jax.Array


def run_ais(slp: SLP, ais: AISConfig, seed: PRNGKey, xs: Trace, lp: jax.Array, N: int):
    assert ais.tempering_schedule[0] == 0.  # log prior
    assert ais.tempering_schedule[-1] == 1. # log joint

    @jax.jit
    def ais_log_weight(x_init: Trace, lp_init: float, rng_key: PRNGKey):
        prior_key, tempering_key = jax.random.split(rng_key)
        prior_kernel_init = broadcast_jaxtree(InferenceCarry(jnp.array(0,int), jnp.array(0,float), InferenceState(x_init, lp_init), []), (1,))
        prior_keys = jax.random.split(prior_key, ais.n_prior_mcmc_steps)
        last_prior_state, _ = jax.lax.scan(ais.prior_kernel, prior_kernel_init, prior_keys)
        # jax.debug.print("last_prior_state {x}", x=last_prior_state)

        def tempering_step(carry: Tuple[InferenceCarry,float], tempering: Tuple[PRNGKey,float]) -> Tuple[Tuple[InferenceCarry,float], None]:
            inference_carry_from_prev_kernel, current_log_weight = carry
            tempering_key, tempering_beta = tempering

            # log weights with respect to current tempering
            log_prior_before, log_likelihood_before, _ = jax.vmap(slp._log_prior_likeli_pathcond)(inference_carry_from_prev_kernel.state.position)
            tempered_log_prob_before_step = log_prior_before + tempering_beta * log_likelihood_before

            current_inference_carry = InferenceCarry(
                inference_carry_from_prev_kernel.iteration,
                jax.lax.broadcast(tempering_beta, (1,)),
                InferenceState(inference_carry_from_prev_kernel.state.position, tempered_log_prob_before_step),
                []
            )

            next_inference_carry, _ = ais.tempering_kernel(current_inference_carry, tempering_key)
            tempered_log_prob_after_step = next_inference_carry.state.log_prob

            next_log_weight = current_log_weight + tempered_log_prob_before_step - tempered_log_prob_after_step
        
            # jax.debug.print("current inference carry {y}\nnext inference carry {x}", x=next_inference_carry, y=current_inference_carry)

            return (next_inference_carry, next_log_weight), None
        
        tempering_keys = jax.random.split(tempering_key, ais.tempering_schedule.size)
        
        # since ais.tempering_schedule[0] == 0. (first tempered distribution is prior) we can input last_prior_state
        (last_tempering_state, log_weight), _ = jax.lax.scan(tempering_step, (last_prior_state, jnp.array([0.], float)), (tempering_keys, ais.tempering_schedule))
        # jax.debug.print("last_tempering_state {x}", x=last_tempering_state)

        # since ais.tempering_schedule[-1] == 1. (last tempered distribution is joint)
        log_prior = last_prior_state.state.log_prob
        log_joint = last_tempering_state.state.log_prob

        return log_weight + log_joint - log_prior

    # return jax.lax.map(lambda arg: ais_log_weight(*arg), (xs, lp, jax.random.split(seed, N)), batch_size=10_000)
    return jax.vmap(ais_log_weight)(xs, lp, jax.random.split(seed, N))
