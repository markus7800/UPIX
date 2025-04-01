import jax
from typing import Callable, Dict, Optional, Tuple
from ..types import PRNGKey, Trace
import dccxjax.distributions as dist
from .mcmc import InferenceState, InferenceAlgorithm, Kernel
from dataclasses import dataclass
import math
from jax.flatten_util import ravel_pytree
from ..utils import maybe_jit_warning, to_shaped_arrays
from .gibbs_model import GibbsModel
from abc import ABC, abstractmethod

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
        return dist.Normal(X, scale) # type: ignore
    return _gaussian

def rw_kernel(
    rng_key: PRNGKey,
    current_state: InferenceState,
    log_prob_fn: Callable[[Trace], float],
    proposer: Callable[[jax.Array], dist.Distribution]
) -> InferenceState:
    
    current_value_flat, unravel_fn = ravel_pytree(current_state.position)
    proposal_dist = proposer(current_value_flat)
    proposal_key, accept_key = jax.random.split(rng_key)
    proposed_value_flat = proposal_dist.sample(proposal_key)
    proposed_value = unravel_fn(proposed_value_flat)
    proposed_log_prob = log_prob_fn(proposed_value)
    
    backward_dist = proposer(proposed_value_flat)
    Q = backward_dist.log_prob(current_value_flat).sum() - proposal_dist.log_prob(proposed_value_flat).sum()
    P = proposed_log_prob - current_state.log_prob

    accept = jax.lax.log(jax.random.uniform(accept_key)) < (P + Q)
    new_state = jax.lax.cond(accept, lambda _: InferenceState(current_state.iteration, proposed_value, proposed_log_prob), lambda _: current_state, operand=None)
    return new_state

def rw_kernel_sparse(   
    rng_key: PRNGKey,
    current_state: InferenceState,
    log_prob_fn: Callable[[Trace], float],
    proposer: Callable[[jax.Array], dist.Distribution],
    p: float # static
) -> InferenceState:
    
    current_value_flat, unravel_fn = ravel_pytree(current_state.position)

    def step(current_value_flat: jax.Array, current_log_prob: jax.Array, step_key: PRNGKey):
        proposal_dist = proposer(current_value_flat)
        proposal_key, mask_key, accept_key = jax.random.split(step_key,3)
        proposed_value_flat = proposal_dist.sample(proposal_key)

        mask = jax.random.bernoulli(mask_key, p, proposed_value_flat.shape)
        proposed_value_flat = jax.lax.select(mask, proposed_value_flat, current_value_flat)

        proposed_value = unravel_fn(proposed_value_flat)
        proposed_log_prob = log_prob_fn(proposed_value)

        backward_dist = proposer(proposed_value_flat)
        Q = backward_dist.log_prob(current_value_flat).sum() - proposal_dist.log_prob(proposed_value_flat).sum()
        P = proposed_log_prob - current_log_prob

        accept = jax.lax.log(jax.random.uniform(accept_key)) < (P + Q)
        new_value_flat = jax.lax.select(accept, proposed_value_flat, current_value_flat)
        new_log_prob = jax.lax.select(accept, proposed_log_prob, current_log_prob)

        return (new_value_flat, new_log_prob), None
    
    scan_keys = jax.random.split(rng_key, int(math.ceil(1./p)))
    (last_position_flat, last_log_prob), _ = jax.lax.scan(lambda c, s : step(*c, s), (current_value_flat, current_state.log_prob), scan_keys) # type: ignore TODO


    return InferenceState(current_state.iteration, unravel_fn(last_position_flat), last_log_prob)  # type: ignore TODO
    

def rw_kernel_elementwise(
    rng_key: PRNGKey,
    current_state: InferenceState,
    log_prob_fn: Callable[[Trace], float],
    proposer: Callable[[jax.Array], dist.Distribution],
    N: int
) -> InferenceState:
    
    def _body(i: int, current_position: Trace, current_log_prob: float, body_rng_key):
        current_value_flat, unravel_fn = ravel_pytree(current_position)
        sub_current_value_flat = current_value_flat[i]
        proposal_dist = proposer(sub_current_value_flat)
        body_rng_key, proposal_key = jax.random.split(body_rng_key)
        sub_proposed_value_flat = proposal_dist.sample(proposal_key)
        proposed_value_flat = current_value_flat.at[i].set(sub_proposed_value_flat)
        proposed_value = unravel_fn(proposed_value_flat)
        proposed_log_prob = log_prob_fn(proposed_value)

        backward_dist = proposer(sub_proposed_value_flat)
        Q = backward_dist.log_prob(sub_current_value_flat).sum() - proposal_dist.log_prob(sub_proposed_value_flat).sum()
        P = proposed_log_prob - current_log_prob

        body_rng_key, accept_key = jax.random.split(body_rng_key)
        accept = jax.lax.log(jax.random.uniform(accept_key)) < (P + Q)

        return jax.lax.cond(accept, lambda _: (proposed_value, proposed_log_prob, body_rng_key), lambda _: (current_position, current_log_prob, body_rng_key), operand=None)


    new_position, new_log_prob, _ = jax.lax.fori_loop(0, N, lambda i, a: _body(i, *a), (current_state.position, current_state.log_prob, rng_key))
    
    return InferenceState(current_state.iteration, new_position, new_log_prob)
    
class RandomWalk(InferenceAlgorithm):
    def __init__(self,
                 proposer: Callable[[jax.Array],dist.Distribution],
                 block_update: bool = True,
                 sparse_frac: Optional[float] = None,
                 sparse_numvar: Optional[int] = None
                 ) -> None:
        
        self.proposer = proposer
        self.jitted_kernel = False
        self.block_update = block_update
        self.sparse_frac = sparse_frac
        self.sparse_numvar = sparse_numvar

    def make_kernel(self, gibbs_model: GibbsModel, step_number: int) -> Kernel:
        if not self.block_update:
            X, _ = gibbs_model.split_trace(gibbs_model.slp.decision_representative)
            assert all(values.shape == () for _, values in X.items())

        sparse = self.sparse_frac is not None or self.sparse_numvar is not None
        sparse_p = 0.
        if sparse:
            assert self.block_update
            assert (self.sparse_frac is None) ^ (self.sparse_numvar is None)
            if self.sparse_frac is not None:
                sparse_p = self.sparse_frac
            if self.sparse_numvar is not None:
                sparse_p = self.sparse_numvar / len(gibbs_model.slp.decision_representative)            

        @jax.jit
        def _rw_kernel(rng_key: PRNGKey, state: InferenceState) -> InferenceState:
            maybe_jit_warning(self, "jitted_kernel", "_rw_kernel", f"Inference step {step_number}: <RandomWalk at {hex(id(self))}>", to_shaped_arrays(state))
            X, Y = gibbs_model.split_trace(state.position)
            gibbs_model.set_Y(Y)
            current_mh_state = InferenceState(state.iteration, X, state.log_prob)
            if self.block_update:
                if sparse:
                    next_mh_state = rw_kernel_sparse(rng_key, current_mh_state, gibbs_model.log_prob, self.proposer, sparse_p)
                else:
                    next_mh_state = rw_kernel(rng_key, current_mh_state, gibbs_model.log_prob, self.proposer)
            else:
                next_mh_state = rw_kernel_elementwise(rng_key, current_mh_state, gibbs_model.log_prob, self.proposer, len(gibbs_model.variables))


            next_mh_state = InferenceState(state.iteration, gibbs_model.combine_to_trace(next_mh_state.position, Y), next_mh_state.log_prob)
            return next_mh_state
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
    current_state: InferenceState,
    log_prob_fn: Callable[[Trace], jax.Array],
    proposal: TraceProposal,
    Y: Trace
) -> InferenceState:

    rng_key, proposal_key = jax.random.split(rng_key)
    proposed_position, foward_lp = proposal.propose(proposal_key, current_state.position | Y)
    backward_lp = proposal.assess(proposed_position | Y, current_state.position)
    Q = backward_lp - foward_lp
    
    proposed_log_prob = log_prob_fn(proposed_position)
    P = proposed_log_prob - current_state.log_prob

    rng_key, accept_key = jax.random.split(rng_key)
    accept = jax.lax.log(jax.random.uniform(accept_key)) < (P + Q)
    jax.debug.print("accept = {a}", a=accept)
    new_state = jax.lax.cond(accept, lambda _: InferenceState(current_state.iteration, proposed_position, proposed_log_prob), lambda _: current_state, operand=None)
    return new_state

class MetropolisHastings(InferenceAlgorithm):
    def __init__(self, proposal: TraceProposal) -> None:
        self.proposal = proposal
        self.jitted_kernel = False

    def make_kernel(self, gibbs_model: GibbsModel, step_number: int) -> Kernel:
        @jax.jit
        def _mh_kernel(rng_key: PRNGKey, state: InferenceState) -> InferenceState:
            maybe_jit_warning(self, "jitted_kernel", "_mh_kernel", f"Inference step {step_number}: <MetropolisHastings at {hex(id(self))}>", to_shaped_arrays(state))
            X, Y = gibbs_model.split_trace(state.position)
            gibbs_model.set_Y(Y)
            current_mh_state = InferenceState(state.iteration, X, state.log_prob)
            next_mh_state = mh_kernel(rng_key, current_mh_state, gibbs_model.log_prob, self.proposal, gibbs_model.Y)
            next_state = InferenceState(state.iteration, gibbs_model.combine_to_trace(next_mh_state.position, Y), next_mh_state.log_prob)
            return next_state
        
            # current_mh_state = InferenceState(state.iteration, state.position, state.log_prob)
            # next_mh_state = mh_kernel(rng_key, current_mh_state, gibbs_model.slp.log_prob, self.proposal)
            # return next_mh_state
         
        return _mh_kernel

MH = MetropolisHastings