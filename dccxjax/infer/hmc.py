import jax
import jax.numpy as jnp
from typing import Callable, Dict, Optional, Tuple, NamedTuple
from ..types import PRNGKey, Trace, FloatArray, BoolArray, IntArray
import dccxjax.distributions as dist
from .mcmc import MCMCState, InferenceInfo, KernelState, MCMCInferenceAlgorithm, Kernel
from multipledispatch import dispatch
from jax.flatten_util import ravel_pytree
from .gibbs_model import GibbsModel
from ..utils import JitVariationTracker, maybe_jit_warning, to_shaped_arrays

__all__ = [
    "HamiltonianMonteCarlo",
    "HMC",
]
class HMCInfo(NamedTuple):
    accepted: IntArray
    diverged: IntArray
    proposed_out_of_support: IntArray

@dispatch(HMCInfo, int)
def summarise_mcmc_info(info: HMCInfo, n_samples: int) -> str:
    n_chains = info.accepted.shape[0]
    acceptance_rate = info.accepted / n_samples
    if n_chains > 10:
        acceptance_rate_mean = jnp.mean(acceptance_rate)
        acceptance_rate_std = jnp.std(acceptance_rate)
        divergences_mean = jnp.mean(info.diverged)
        divergences_std = jnp.std(info.diverged)
        frac = info.proposed_out_of_support / jnp.maximum(info.diverged,1.0)
        out_of_support_mean = jnp.mean(frac)
        out_of_support_std = jnp.std(frac)
        
        s = f"HMCInfo for {n_chains} chains - acceptance rates: {acceptance_rate_mean.item():.4f} +/- {acceptance_rate_std.item():.4f}, "
        s += f"divergences (% out-of-support): {divergences_mean.item():.4f} +/- {divergences_std.item():.4f} ({out_of_support_mean.item():.4f} +/- {out_of_support_std.item():.4f})"
        return s
    else:
        s = f"HMCInfo for {n_chains} chains - acceptance rates: [" + ", ".join([f"{ar.item():.4f}" for ar in acceptance_rate]) + "], "
        s += f"divergences (% out-of-support): [" + ", ".join([f"{d.item():,} ({o.item()/max(d.item(),1):.2f})" for d, o in zip(info.diverged,info.proposed_out_of_support)]) + "]"
        return s

class LeapfrogState(NamedTuple):
    x: FloatArray
    p: FloatArray

def hmc_kernel(
    rng_key: PRNGKey,
    current_position: FloatArray,
    current_log_prob: FloatArray,
    log_prob_fn: Callable[[FloatArray], FloatArray],
    L: int,
    eps: float,
    debug: bool
    ) -> Tuple[FloatArray,FloatArray,BoolArray,BoolArray]:
    
    proposal_key, accept_key = jax.random.split(rng_key)

    p_current = jax.random.normal(proposal_key, shape=current_position.shape)
    k_current = p_current.dot(p_current) / 2

    # leapfrog integrator

    grad_log_prob_fn: Callable[[FloatArray], FloatArray] = jax.grad(log_prob_fn)
    
    # half step
    x = current_position
    p = p_current

    p = p - eps/2 * -grad_log_prob_fn(x)
    if debug: jax.debug.print("x={x} p={p}", x=x,p=p)

    # L-1 full steps
    def leapfrog_step(state: LeapfrogState, _) -> Tuple[LeapfrogState, None]:
        x_new = state.x + eps * state.p
        p_new = state.p - eps * -grad_log_prob_fn(x_new)
        if debug: jax.debug.print("x={x} p={p}", x=x_new,p=p_new)
        return LeapfrogState(x_new, p_new), None
    
    (x, p), _ = jax.lax.scan(leapfrog_step, LeapfrogState(x, p), length=L-1)

    # half step
    x = x + eps * p
    p = p - eps/2 * -grad_log_prob_fn(x)
    if debug: jax.debug.print("x={x} p={p}", x=x,p=p)

    # leapfrog finished

    proposed_position = x
    proposed_log_prob = log_prob_fn(proposed_position)
    p_proposed = -p
    k_proposed = p_proposed.dot(p_proposed) / 2

    energy_current = -current_log_prob + k_current
    energy_proposed = -proposed_log_prob + k_proposed

    energy_delta = energy_current - energy_proposed
    energy_delta = jax.lax.select(jnp.isnan(energy_delta), -jnp.inf, energy_delta)
    diverged: BoolArray = -energy_delta > 1000

    accept: BoolArray = jax.lax.log(jax.random.uniform(accept_key)) < energy_delta

    if debug: jax.debug.print("proposed_log_prob={a} k_proposed={b} current_log_prob={c}, k_current={d}", a=proposed_log_prob, b=k_proposed, c=current_log_prob, d=k_current)
    if debug: jax.debug.print("energy_current={a} energy_proposed={b} diverged={c}, accept={d}", a=energy_current, b=energy_proposed, c=diverged, d=accept)


    return proposed_position, proposed_log_prob, accept, diverged

from functools import partial

class HamiltonianMonteCarlo(MCMCInferenceAlgorithm):
    def __init__(self,
                 L: int,
                 eps: float,
                 unconstrained: bool = False
                 ) -> None:
        self.L = L
        self.eps = eps
        self.unconstrained = unconstrained


    def init_info(self) -> InferenceInfo:
        return HMCInfo(jnp.array(0,int), jnp.array(0,int), jnp.array(0,int))
    
    def make_kernel(self, gibbs_model: GibbsModel, step_number: int, collect_inferenence_info: bool) -> Kernel:
        X_repr, _ = gibbs_model.split_trace(gibbs_model.slp.decision_representative)
        is_discrete_map = gibbs_model.slp.get_is_discrete_map()

        assert all(not is_discrete_map[addr] for addr, _ in X_repr.items())


        jit_tracker = JitVariationTracker(f"_hmc_kernel for Inference step {step_number}: <HMC at {hex(id(self))}>")
        @partial(jax.jit, static_argnums=[3])
        def _hmc_kernel(rng_key: PRNGKey, temperature: FloatArray, state: KernelState, debug: bool=False) -> KernelState:
            maybe_jit_warning(jit_tracker, str(to_shaped_arrays((temperature, state))))
            
            current_postion = gibbs_model.slp.transform_to_unconstrained(state.position) if self.unconstrained else state.position

            X, Y = gibbs_model.split_trace(current_postion)
            gibbs_model.set_Y(Y)
        
            X_flat, unravel_fn = ravel_pytree(X)
            if self.unconstrained:
                _tempered_log_prob_fn = gibbs_model.unraveled_unconstrained_tempered_log_prob(temperature, unravel_fn)
            else:
                _tempered_log_prob_fn = gibbs_model.unraveled_tempered_log_prob(temperature, unravel_fn)

            lp = _tempered_log_prob_fn(X_flat) if self.unconstrained else state.log_prob

            proposed_X, proposed_log_prob, accept, diverged = hmc_kernel(rng_key, X_flat, lp, _tempered_log_prob_fn, self.L, self.eps, debug)
            next_X_flat, next_log_prob = jax.lax.cond(accept, lambda _: (proposed_X, proposed_log_prob), lambda _: (X_flat, state.log_prob), operand=None)
            
            
            next_position = gibbs_model.combine_to_trace(unravel_fn(next_X_flat), Y)
            if self.unconstrained:
                next_position = gibbs_model.slp.transform_to_constrained(next_position)
                next_log_prob = gibbs_model.tempered_log_prob(temperature)(next_position)

            hmc_info = state.info
            if collect_inferenence_info:
                assert isinstance(hmc_info, HMCInfo)
                hmc_info = HMCInfo(
                    hmc_info.accepted + accept,
                    hmc_info.diverged + diverged,
                    hmc_info.proposed_out_of_support + (jnp.isinf(proposed_log_prob))
                )

            return KernelState(next_position, next_log_prob, hmc_info)
        
        return _hmc_kernel
    
HMC = HamiltonianMonteCarlo