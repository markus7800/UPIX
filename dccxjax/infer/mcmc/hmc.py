import jax
import jax.numpy as jnp
from typing import Callable, Dict, Optional, Tuple, NamedTuple
from dccxjax.types import PRNGKey, Trace, FloatArray, BoolArray, IntArray
from dccxjax.infer.mcmc.mcmc_core import MCMCState, InferenceInfo, KernelState, MCMCInferenceAlgorithm, Kernel, AnnealingMask
from multipledispatch import dispatch
from jax.flatten_util import ravel_pytree
from dccxjax.infer.gibbs_model import GibbsModel, SLP
from dccxjax.utils import JitVariationTracker, maybe_jit_warning, pprint_dtype_shape_of_tree
from dccxjax.infer.variable_selector import VariableSelector, AllVariables

__all__ = [
    "HamiltonianMonteCarlo",
    "HMC",
    "DiscontinuousHamiltonianMonteCarlo",
    "DHMC",
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

class LeapfrogState2(NamedTuple):
    x: FloatArray
    g: FloatArray # gradient of U at x
    p: FloatArray

def hmc_kernel(
    rng_key: PRNGKey,
    current_position: FloatArray,
    current_log_prob: FloatArray,
    log_prob_fn: Callable[[FloatArray], FloatArray],
    L: int,
    eps: float
    ) -> Tuple[FloatArray,FloatArray,BoolArray,BoolArray]:
    
    proposal_key, accept_key = jax.random.split(rng_key)

    p_current = jax.random.normal(proposal_key, shape=current_position.shape)
    k_current = p_current.dot(p_current) / 2

    # leapfrog integrator

    grad_log_prob_fn: Callable[[FloatArray], FloatArray] = jax.grad(log_prob_fn)
    
    # initial
    x = current_position
    p = p_current

    # half step on momentum
    p = p + eps/2 * grad_log_prob_fn(x)

    # L-1 full steps on position and momentum
    def leapfrog_step(state: LeapfrogState, _) -> Tuple[LeapfrogState, None]:
        x_new = state.x + eps * state.p
        p_new = state.p + eps * grad_log_prob_fn(x_new) # grad_U = -grad_log_prob_fn
        return LeapfrogState(x_new, p_new), None
    
    (x, p), _ = jax.lax.scan(leapfrog_step, LeapfrogState(x, p), length=L-1)

    # full step on position and half step on momentum
    x = x + eps * p
    p = p + eps/2 * grad_log_prob_fn(x)

    # leapfrog finished

    # more compactly, but more ops and have to store grad
    # L steps
    # def leapfrog_step2(state: LeapfrogState2, _) -> Tuple[LeapfrogState2, None]:
    #     p_half_step = state.p - eps / 2 * state.g
    #     x_new = state.x + eps * p_half_step
    #     g_new = -grad_log_prob_fn(x_new)
    #     p_new = p_half_step - eps / 2 * g_new
    #     return LeapfrogState2(x_new, g_new, p_new), None
    
    # (x, _, p), _ = jax.lax.scan(leapfrog_step2, LeapfrogState2(x, -grad_log_prob_fn(x), p), length=L)

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

    return proposed_position, proposed_log_prob, accept, diverged


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
    
    def __repr__(self) -> str:
        s = f"HMC(L={self.L}, eps={self.eps}, at {hex(id(self))})"
        return s

    def make_kernel(self, gibbs_model: GibbsModel, step_number: int, collect_inferenence_info: bool) -> Kernel:
        X_repr, _ = gibbs_model.split_trace(gibbs_model.slp.decision_representative)
        is_discrete_map = gibbs_model.slp.get_is_discrete_map()
        assert all(not is_discrete_map[addr] for addr in X_repr.keys()), "No discrete parameters allowed in HMC."

        jit_tracker = JitVariationTracker(f"_hmc_kernel for Inference step {step_number}: <HMC at {hex(id(self))}>")
        @jax.jit
        def _hmc_kernel(rng_key: PRNGKey, temperature: FloatArray, data_annealing: AnnealingMask, state: KernelState) -> KernelState:
            maybe_jit_warning(jit_tracker, str(pprint_dtype_shape_of_tree((temperature, state))))
            # jax.debug.print("key={k}", k=rng_key)
            
            X_flat, log_prob, unravel_fn, target_fn = self.default_preprocess_to_flat(gibbs_model, temperature, data_annealing, state)

            proposed_X, proposed_log_prob, accept, diverged = hmc_kernel(rng_key, X_flat, log_prob, target_fn, self.L, self.eps)
            next_X_flat, next_log_prob = jax.lax.cond(accept, lambda _: (proposed_X, proposed_log_prob), lambda _: (X_flat, log_prob), operand=None)
            
            next_stats = self.default_postprocess_from_flat(gibbs_model, next_X_flat, next_log_prob, unravel_fn)

            hmc_info = state.info
            if collect_inferenence_info:
                assert isinstance(hmc_info, HMCInfo)
                hmc_info = HMCInfo(
                    hmc_info.accepted + accept,
                    hmc_info.diverged + diverged,
                    hmc_info.proposed_out_of_support + (jnp.isinf(proposed_log_prob))
                )

            return KernelState(next_stats, hmc_info)
        
        return _hmc_kernel
    
HMC = HamiltonianMonteCarlo

class DiscontLeapfrogState(NamedTuple):
    x: FloatArray
    u: FloatArray
    g: FloatArray # gradient of U at x
    p: FloatArray

class CoordIntegratorState(NamedTuple):
    x: FloatArray
    u: FloatArray
    p: FloatArray

from jax._src.random import _shuffle, _check_prng_key
def dhmc_kernel(
    rng_key: PRNGKey,
    current_position: FloatArray,
    current_log_prob: FloatArray,
    log_prob_fn: Callable[[FloatArray], FloatArray],
    L: int,
    eps_min: float,
    eps_max: float,
    all_discontinuous: bool,
    discontinuous_mask: BoolArray, # we have at least on discontinuous
    discontinuous_ixs: IntArray
    ) -> Tuple[FloatArray,FloatArray,BoolArray,BoolArray]:
    
    proposal_key, eps_key, permute_key, accept_key = jax.random.split(rng_key,4)

    p_current = jax.lax.select(discontinuous_mask,
                               jax.random.laplace(proposal_key, shape=current_position.shape), 
                               jax.random.normal(proposal_key, shape=current_position.shape))
    k_current = jax.lax.select(discontinuous_mask, jax.lax.abs(p_current), jax.lax.square(p_current)/2).sum()

    eps = jax.random.uniform(eps_key, minval=eps_min, maxval=eps_max)

    # mixed continuous / discontinuous leapfrog integrator

    grad_log_prob_fn: Callable[[FloatArray], FloatArray] = jax.grad(log_prob_fn)

    # initial
    x = current_position
    p = p_current


    def coord_integrator(state: CoordIntegratorState, ix: IntArray) -> Tuple[CoordIntegratorState, None]:
        x, u, p = state
        x_new = x.at[ix].add(eps * jax.lax.sign(p[ix]))
        u_new = -log_prob_fn(x_new)
        delta_u = u_new - u
        p_new = p.at[ix].add(-jax.lax.sign(p[ix]) * delta_u)
        accept = jax.lax.abs(p[ix]) > delta_u
        
        # cond here causes some issues with shard_map for some reason
        new_state = jax.lax.cond(accept,
            lambda _: CoordIntegratorState(x_new, u_new, p_new),
            lambda _: CoordIntegratorState(x, u, -p),
            operand=None
        )
        return new_state, None
        
        # this works with shard_map
        x_next = jax.lax.select(accept, x_new, x)
        u_next = jax.lax.select(accept, u_new, u)
        p_next = jax.lax.select(accept, p_new, -p)
        return CoordIntegratorState(x_next, u_next, p_next), None
        
    
    def leapfrog_step(state: DiscontLeapfrogState, permute_key: PRNGKey) -> Tuple[DiscontLeapfrogState, None]:
        x, u, g, p = state

        if not all_discontinuous:
            # leave discontinuous variables untouched
            p_half_step = jax.lax.select(discontinuous_mask, p, p - eps / 2 * g)
            x_half_step = jax.lax.select(discontinuous_mask, x, x + eps / 2 * p_half_step)
            
            x = x_half_step
            u = -log_prob_fn(x_half_step)
            # do not need to update g here
            p = p_half_step


        # ixs = jax.random.permutation(permute_key, discontinuous_ixs) # bad for sharding
        # ixs = jax.random.choice(permute_key, discontinuous_ixs, (len(discontinuous_ixs),), replace=False) # uses permutation under the hood
        ixs = jax.random.choice(permute_key, discontinuous_ixs, (len(discontinuous_ixs),), replace=True) # this is ok
        # ixs = _shuffle(_check_prng_key("", permute_key)[0], discontinuous_ixs, 0)
        # ixs = discontinuous_ixs

        (x, u, p), _ = jax.lax.scan(coord_integrator, CoordIntegratorState(x, u, p), ixs) # makes gradient g invalid

        if not all_discontinuous:
            # leave discontinuous variables untouched
            x = jax.lax.select(discontinuous_mask, x, x + eps / 2 * p)
            u = -log_prob_fn(x)
            g = -grad_log_prob_fn(x)
            p = jax.lax.select(discontinuous_mask, p, p - eps / 2 * g)
        # else gradient g is not updated

        return DiscontLeapfrogState(x, u, g, p), None
    
    (x, u, _, p), _ = jax.lax.scan(
        leapfrog_step,
        DiscontLeapfrogState(x, -log_prob_fn(x), -grad_log_prob_fn(x) if not all_discontinuous else jnp.array(0,float), p),
        jax.random.split(permute_key,L))
    
    # leapfrog finished

    proposed_position = x
    proposed_log_prob = -u
    p_proposed = -p
    k_proposed = jax.lax.select(discontinuous_mask, jax.lax.abs(p_proposed), jax.lax.square(p_proposed)/2).sum()

    energy_current = -current_log_prob + k_current
    energy_proposed = -proposed_log_prob + k_proposed

    energy_delta = energy_current - energy_proposed
    energy_delta = jax.lax.select(jnp.isnan(energy_delta), -jnp.inf, energy_delta)
    diverged: BoolArray = -energy_delta > 1000

    accept: BoolArray = jax.lax.log(jax.random.uniform(accept_key)) < energy_delta

    return proposed_position, proposed_log_prob, accept, diverged

class DiscontinuousHamiltonianMonteCarlo(MCMCInferenceAlgorithm):
    def __init__(self,
                 L: int,
                 eps_min: float,
                 eps_max: float,
                 discontinuous: VariableSelector = AllVariables(),
                 unconstrained: bool = False,
                 ) -> None:
        self.L = L
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.unconstrained = unconstrained
        self.discontinuous = discontinuous

    def init_info(self) -> InferenceInfo:
        return HMCInfo(jnp.array(0,int), jnp.array(0,int), jnp.array(0,int))
    
    def __repr__(self) -> str:
        s = f"DHMC(L={self.L}, eps=[{self.eps_min},{self.eps_max}], discont={self.discontinuous}, at {hex(id(self))})"
        return s

    def make_kernel(self, gibbs_model: GibbsModel, step_number: int, collect_inferenence_info: bool) -> Kernel:
        X_repr, _ = gibbs_model.split_trace(gibbs_model.slp.decision_representative)
        is_discrete_map = gibbs_model.slp.get_is_discrete_map()
        assert all(not is_discrete_map[addr] for addr in X_repr.keys()), "No discrete parameters allowed in DHMC."
        X_repr_is_discontinuous = {addr: self.discontinuous.contains(addr) for addr in X_repr.keys()}
        all_discontinuous = all(X_repr_is_discontinuous.values())
        any_discontinuous = any(X_repr_is_discontinuous.values())

        if not any_discontinuous:
            return HMC(self.L, (self.eps_min + self.eps_max) / 2, self.unconstrained).make_kernel(gibbs_model, step_number, collect_inferenence_info)

        discontinuous_mask, _ = ravel_pytree(X_repr_is_discontinuous)
        discontinuous_ixs = jax.lax.pvary(jnp.arange(0,discontinuous_mask.shape[0],dtype=int)[discontinuous_mask.astype(bool)], ("i",))
        # print("discontinuous_ixs:", discontinuous_ixs)

        jit_tracker = JitVariationTracker(f"_dhmc_kernel for Inference step {step_number}: <DHMC at {hex(id(self))}>")
        @jax.jit
        def _dhmc_kernel(rng_key: PRNGKey, temperature: FloatArray, data_annealing: AnnealingMask, state: KernelState) -> KernelState:
            maybe_jit_warning(jit_tracker, str(pprint_dtype_shape_of_tree((temperature, state))))
            
            X_flat, log_prob, unravel_fn, target_fn = self.default_preprocess_to_flat(gibbs_model, temperature, data_annealing, state)

            proposed_X, proposed_log_prob, accept, diverged = dhmc_kernel(
                rng_key, X_flat, log_prob, target_fn, self.L, self.eps_min, self.eps_max,
                all_discontinuous, discontinuous_mask, discontinuous_ixs
            )
            next_X_flat, next_log_prob = jax.lax.cond(accept, lambda _: (proposed_X, proposed_log_prob), lambda _: (X_flat, log_prob), operand=None)
            
            next_stats = self.default_postprocess_from_flat(gibbs_model, next_X_flat, next_log_prob, unravel_fn)

            hmc_info = state.info
            if collect_inferenence_info:
                assert isinstance(hmc_info, HMCInfo)
                hmc_info = HMCInfo(
                    hmc_info.accepted + accept,
                    hmc_info.diverged + diverged,
                    hmc_info.proposed_out_of_support + (jnp.isinf(proposed_log_prob))
                )

            return KernelState(next_stats, hmc_info)
        
        return _dhmc_kernel
    
DHMC = DiscontinuousHamiltonianMonteCarlo