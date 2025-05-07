import jax
import jax.numpy as jnp
from typing import Callable, Dict, Optional, Tuple, NamedTuple
from ..types import PRNGKey, Trace, FloatArray, BoolArray, IntArray
import dccxjax.distributions as dist
from .mcmc import MCMCState, InferenceInfo, KernelState, MCMCInferenceAlgorithm, Kernel
from dataclasses import dataclass
import math
from jax.flatten_util import ravel_pytree
from ..utils import JitVariationTracker, maybe_jit_warning, to_shaped_arrays
from .gibbs_model import GibbsModel
from abc import ABC, abstractmethod
from multipledispatch import dispatch


__all__ = [
    "gaussian_random_walk",
    "RandomWalk",
    "RW",
    "TraceProposal",
    "MetropolisHastings",
    "MH",
]


def gaussian_random_walk(scale: float):
    def _gaussian(X: jax.Array) -> dist.Distribution:
        return dist.Normal(X, scale)
    return _gaussian

def rw_kernel(
    rng_key: PRNGKey,
    current_position: Trace,
    current_log_prob: FloatArray,
    log_prob_fn: Callable[[Trace], FloatArray],
    proposer: Callable[[jax.Array], dist.Distribution]
) -> Tuple[Trace, FloatArray, BoolArray]:
    
    current_value_flat, unravel_fn = ravel_pytree(current_position)
    proposal_dist = proposer(current_value_flat)
    proposal_key, accept_key = jax.random.split(rng_key)
    proposed_value_flat = proposal_dist.sample(proposal_key)
    proposed_position = unravel_fn(proposed_value_flat)
    proposed_log_prob = log_prob_fn(proposed_position)
    
    backward_dist = proposer(proposed_value_flat)
    Q = backward_dist.log_prob(current_value_flat).sum() - proposal_dist.log_prob(proposed_value_flat).sum()
    P = proposed_log_prob - current_log_prob

    accept = jax.lax.log(jax.random.uniform(accept_key)) < (P + Q)
    new_position, new_log_prob = jax.lax.cond(accept, lambda _: (proposed_position, proposed_log_prob), lambda _: (current_position, current_log_prob), operand=None)
    return new_position, new_log_prob, accept

def rw_kernel_sparse(   
    rng_key: PRNGKey,
    current_position: Trace,
    current_log_prob: FloatArray,
    log_prob_fn: Callable[[Trace], FloatArray],
    proposer: Callable[[jax.Array], dist.Distribution],
    p: float, # static
    n_dim: int
) -> Tuple[Trace, FloatArray, IntArray]:
    
    current_value_flat, unravel_fn = ravel_pytree(current_position)
    assert current_value_flat.shape == (n_dim,)

    def step(current_value_flat: FloatArray, current_log_prob: FloatArray, n_accepted: IntArray, step_key: PRNGKey) -> Tuple[Tuple[FloatArray,FloatArray,IntArray],None]:
        proposal_dist = proposer(current_value_flat)
        proposal_key, mask_key, accept_key = jax.random.split(step_key,3)
        proposed_value_flat = proposal_dist.sample(proposal_key)

        mask = jax.random.bernoulli(mask_key, p, proposed_value_flat.shape)
        proposed_value_flat = jax.lax.select(mask, proposed_value_flat, current_value_flat)

        proposed_position = unravel_fn(proposed_value_flat)
        proposed_log_prob = log_prob_fn(proposed_position)

        backward_dist = proposer(proposed_value_flat)
        Q = backward_dist.log_prob(current_value_flat).sum() - proposal_dist.log_prob(proposed_value_flat).sum()
        P = proposed_log_prob - current_log_prob

        accept = jax.lax.log(jax.random.uniform(accept_key)) < (P + Q)
        new_value_flat = jax.lax.select(accept, proposed_value_flat, current_value_flat)
        new_log_prob = jax.lax.select(accept, proposed_log_prob, current_log_prob)

        return (new_value_flat, new_log_prob, n_accepted + accept), None
    
    scan_keys = jax.random.split(rng_key, int(math.ceil(1./p)))
    (last_position_flat, last_log_prob, n_accepted), _ = jax.lax.scan(lambda c, s : step(*c, s), (current_value_flat, current_log_prob, jnp.array(0,int)), scan_keys)


    return unravel_fn(last_position_flat), last_log_prob, n_accepted
    

def rw_kernel_elementwise(
    rng_key: PRNGKey,
    current_position: Trace,
    current_log_prob: FloatArray,
    log_prob_fn: Callable[[Trace], FloatArray],
    proposer: Callable[[jax.Array], dist.Distribution],
    n_dim: int
) -> Tuple[Trace, FloatArray, IntArray]:
    
    def _body(i: int, current_position: Trace, current_log_prob: FloatArray, n_accept: IntArray, body_rng_key: PRNGKey) -> Tuple[FloatArray,FloatArray,IntArray]:
        current_value_flat, unravel_fn = ravel_pytree(current_position)
        assert current_value_flat.shape == (n_dim,)
        sub_current_value_flat = current_value_flat[i]
        proposal_dist = proposer(sub_current_value_flat)
        body_rng_key, proposal_key = jax.random.split(body_rng_key)
        sub_proposed_value_flat = proposal_dist.sample(proposal_key)
        proposed_value_flat = current_value_flat.at[i].set(sub_proposed_value_flat)
        proposed_position = unravel_fn(proposed_value_flat)
        proposed_log_prob = log_prob_fn(proposed_position)

        backward_dist = proposer(sub_proposed_value_flat)
        Q = backward_dist.log_prob(sub_current_value_flat).sum() - proposal_dist.log_prob(sub_proposed_value_flat).sum()
        P = proposed_log_prob - current_log_prob

        body_rng_key, accept_key = jax.random.split(body_rng_key)
        accept = jax.lax.log(jax.random.uniform(accept_key)) < (P + Q)

        return jax.lax.cond(accept, lambda _: (proposed_position, proposed_log_prob, n_accept + 1, body_rng_key), lambda _: (current_position, current_log_prob, n_accept, body_rng_key), operand=None)


    new_position, new_log_prob, n_accepted, _ = jax.lax.fori_loop(0, n_dim, lambda i, a: _body(i, *a), (current_position, current_log_prob, jnp.array(0,int), rng_key))
    
    return new_position, new_log_prob, n_accepted
    
class MHInfo(NamedTuple):
    accepted: jax.Array

@dispatch(MHInfo, int)
def summarise_mcmc_info(info: MHInfo, n_samples: int) -> str:
    n_chains = info.accepted.shape[0]
    acceptance_rate = info.accepted / n_samples
    if n_chains > 10:
        acceptance_rate_mean = jnp.mean(acceptance_rate)
        acceptance_rate_std = jnp.std(acceptance_rate)
        return f"Acceptance rate: {acceptance_rate_mean.item():.4f} +/-  {acceptance_rate_std.item():.4f}"
    else:
        return f"Acceptance rates for {n_chains} chains: [" + ", ".join([f"{ar.item():.4f}" for ar in acceptance_rate]) + "]"

class RandomWalk(MCMCInferenceAlgorithm):
    def __init__(self,
                 proposer: Callable[[jax.Array],dist.Distribution],
                 elementwise: bool = False,
                 sparse_frac: Optional[float] = None,
                 sparse_numvar: Optional[int] = None
                 ) -> None:

        self.proposer = proposer
        self.jitted_kernel = False
        self.elementwise = elementwise
        self.sparse_frac = sparse_frac
        self.sparse_numvar = sparse_numvar
        self.sparse = self.sparse_frac is not None or self.sparse_numvar is not None

    def init_info(self) -> InferenceInfo:
        if not self.elementwise:
            if self.sparse:
                return MHInfo(jnp.array(0,float))
            else:
                return MHInfo(jnp.array(0,int))
        else:
            return MHInfo(jnp.array(0,float))
    
    def make_kernel(self, gibbs_model: GibbsModel, step_number: int, collect_inferenence_info: bool) -> Kernel:
        X_repr, _ = gibbs_model.split_trace(gibbs_model.slp.decision_representative)
        n_dim = sum(values.size for _, values in X_repr.items())
        is_discrete_map = gibbs_model.slp.get_is_discrete_map()
        # for type stability either all discrete or all continuous
        assert all(is_discrete_map[addr] for addr, _ in X_repr.items()) or all(not is_discrete_map[addr] for addr, _ in X_repr.items())

        sparse_p = 0.
        if self.sparse:
            assert not self.elementwise
            assert (self.sparse_frac is None) ^ (self.sparse_numvar is None)
            if self.sparse_frac is not None:
                sparse_p = self.sparse_frac
            if self.sparse_numvar is not None:
                sparse_p = self.sparse_numvar / n_dim

        jit_tracker = JitVariationTracker(f"_rw_kernel for Inference step {step_number}: <RandomWalk at {hex(id(self))}>")
        @jax.jit
        def _rw_kernel(rng_key: PRNGKey, temperature: FloatArray, state: KernelState) -> KernelState:
            maybe_jit_warning(jit_tracker, str(to_shaped_arrays((temperature, state))))
            X, Y = gibbs_model.split_trace(state.position)
            gibbs_model.set_Y(Y)
        
            _tempered_log_prob_fn = gibbs_model.tempered_log_prob(temperature)
            if not self.elementwise:
                if self.sparse:
                    next_X, next_log_prob, n_accepted = rw_kernel_sparse(rng_key, X, state.log_prob, _tempered_log_prob_fn, self.proposer, sparse_p, n_dim)
                    accepted = n_accepted.sum() / int(math.ceil(1./sparse_p))
                else:
                    next_X, next_log_prob, accepted = rw_kernel(rng_key, X, state.log_prob, _tempered_log_prob_fn, self.proposer)
            else:
                next_X, next_log_prob, n_accepted = rw_kernel_elementwise(rng_key, X, state.log_prob, _tempered_log_prob_fn, self.proposer, n_dim)
                accepted = n_accepted.sum() / n_dim


            mh_info = state.info
            if collect_inferenence_info:
                assert isinstance(mh_info, MHInfo)
                mh_info = MHInfo(mh_info.accepted + accepted)

            return KernelState(gibbs_model.combine_to_trace(next_X, Y), next_log_prob, mh_info)
        
        return _rw_kernel
    
RW = RandomWalk


class TraceProposal(ABC):
    @abstractmethod
    def propose(self, rng_key: PRNGKey, current: Trace) -> Tuple[Trace, jax.Array]:
        raise NotImplementedError
    
    @abstractmethod
    def assess(self, current: Trace, proposed: Trace) -> jax.Array:
        raise NotImplementedError

def mh_kernel(
    rng_key: PRNGKey,
    current_position: Trace,
    current_log_prob: FloatArray,
    log_prob_fn: Callable[[Trace], FloatArray],
    proposal: TraceProposal,
    Y: Trace
) -> Tuple[Trace, FloatArray, BoolArray]:

    rng_key, proposal_key = jax.random.split(rng_key)
    proposed_position, foward_lp = proposal.propose(proposal_key, current_position | Y)
    backward_lp = proposal.assess(proposed_position | Y, current_position)
    Q = backward_lp - foward_lp
    
    proposed_log_prob = log_prob_fn(proposed_position)
    P = proposed_log_prob - current_log_prob

    rng_key, accept_key = jax.random.split(rng_key)
    accept = jax.lax.log(jax.random.uniform(accept_key)) < (P + Q)
    new_position, new_log_prob = jax.lax.cond(accept, lambda _: (proposed_position, proposed_log_prob), lambda _: (current_position, current_log_prob), operand=None)
    return new_position, new_log_prob, accept

class MetropolisHastings(MCMCInferenceAlgorithm):
    def __init__(self, proposal: TraceProposal) -> None:
        self.proposal = proposal
        self.jitted_kernel = False

    def init_info(self) -> InferenceInfo:
        return MHInfo(jnp.array(0,int))
    
    def make_kernel(self, gibbs_model: GibbsModel, step_number: int, collect_inferenence_info: bool) -> Kernel:
        jit_tracker = JitVariationTracker(f"_mh_kernel for Inference step {step_number}: <MetropolisHastings at {hex(id(self))}>")
        @jax.jit
        def _mh_kernel(rng_key: PRNGKey, temperature: FloatArray, state: KernelState) -> KernelState:
            maybe_jit_warning(jit_tracker, str(to_shaped_arrays((temperature, state))))
            X, Y = gibbs_model.split_trace(state.position)
            gibbs_model.set_Y(Y)
            # current_mh_state = InferenceState(X, carry.state.log_prob)
            
            next_X, next_log_prob, accepted = mh_kernel(rng_key, X, state.log_prob, gibbs_model.tempered_log_prob(temperature), self.proposal, gibbs_model.Y)

            mh_info = state.info
            if collect_inferenence_info:
                assert isinstance(mh_info, MHInfo)
                mh_info = MHInfo(mh_info.accepted + accepted)

            return KernelState(gibbs_model.combine_to_trace(next_X, Y), next_log_prob, mh_info)
        
        return _mh_kernel
    

MH = MetropolisHastings