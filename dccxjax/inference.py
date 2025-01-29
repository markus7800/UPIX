from .slp_gen import SLP
from .variable_selector import VariableSelector
from .gibbs import GibbsModel
from typing import List, Dict, Optional, Tuple, Generator, Any, NamedTuple, Callable
import jax
from .samplecontext import Model
from abc import ABC, abstractmethod
from .types import PRNGKey, Trace
import numpyro.distributions as dist
from .utils import maybe_jit_warning, to_shaped_arrays

AbstractInferenceState = NamedTuple # ("AbstractInferenceState", [("position", Trace)])
class InferenceState(AbstractInferenceState):
    position: Trace
    
def InferenceStateFromSubclass(state: AbstractInferenceState):
    return InferenceState(position=state.position)
    
Kernel = Callable[[AbstractInferenceState,PRNGKey],AbstractInferenceState]

class InferenceAlgorithm(ABC):
    @abstractmethod
    def make_kernel(self, gibbs_model: GibbsModel, step_number: int) -> Kernel:
        raise NotImplementedError


class MHState(AbstractInferenceState):
    position: Trace
    log_prob: float


def mh_kernel(
    rng_key: PRNGKey,
    current_state: MHState,
    log_prob_fn: Callable[[Trace], float],
    proposer_map: Dict[str, Callable[[jax.Array], dist.Distribution]]
):
    proposed_position: Trace = dict()

    Q = 0.
    for address, value in current_state.position.items():
        proposer = proposer_map[address]
        proposal_dist = proposer(value)
        rng_key, proposal_key = jax.random.split(rng_key)
        proposed_value = proposal_dist.sample(proposal_key)
        proposed_position[address] = proposed_value
        backward_dist = proposer(proposed_value)
        Q += backward_dist.log_prob(value) - proposal_dist.log_prob(proposed_value)
    
    proposed_log_prob = log_prob_fn(proposed_position)
    P = proposed_log_prob - current_state.log_prob

    rng_key, accept_key = jax.random.split(rng_key)
    accept = jax.lax.log(jax.random.uniform(accept_key)) < (P + Q)
    new_state = jax.lax.cond(accept, lambda _: MHState(proposed_position, proposed_log_prob), lambda _: current_state, operand=None)
    return new_state

def gaussian_random_walk(scale: float):
    def _gaussian(X: jax.Array) -> dist.Distribution:
        return dist.Normal(X, scale) # type: ignore
    return _gaussian

# class MetropolisHastings(InferenceAlgorithm):
#     def __init__(self, proposer_map: Dict[VariableSelector,Callable[[jax.Array],dist.Distribution]]) -> None:
#         self.proposer_map = proposer_map

#     def make_kernel(self, gibbs_model: GibbsModel) -> Kernel:
#         proposer_map: Dict[str, Callable[[jax.Array], dist.Distribution]] = {}
#         for address in gibbs_model.variables:
#             for selector, func in self.proposer_map.items():
#                 if selector.contains(address):
#                     proposer_map[address] = func
#                     break

#         def _kernel(X: InferenceState, rng_key: PRNGKey) -> MHState:
#             log_prob = X.log_prob if hasattr(X, "log_prob") else gibbs_model.log_prob(X.position)
#             state = MHState(X.position, log_prob)
#             return mh_kernel(rng_key, state, gibbs_model.log_prob, proposer_map)
        
#         return _kernel

# MH = MetropolisHastings

from jax.flatten_util import ravel_pytree
def rw_kernel(
    rng_key: PRNGKey,
    current_state: MHState,
    log_prob_fn: Callable[[Trace], float],
    proposer: Callable[[jax.Array], dist.Distribution]
):
    current_value_flat, unravel_fn = ravel_pytree(current_state.position)
    proposal_dist = proposer(current_value_flat)
    rng_key, proposal_key = jax.random.split(rng_key)
    proposed_value_flat = proposal_dist.sample(proposal_key)
    proposed_value = unravel_fn(proposed_value_flat)
    proposed_log_prob = log_prob_fn(proposed_value)
    
    backward_dist = proposer(proposed_value_flat)
    Q = backward_dist.log_prob(current_value_flat).sum() - proposal_dist.log_prob(proposed_value_flat).sum()
    P = proposed_log_prob - current_state.log_prob


    rng_key, accept_key = jax.random.split(rng_key)
    accept = jax.lax.log(jax.random.uniform(accept_key)) < (P + Q)
    new_state = jax.lax.cond(accept, lambda _: MHState(proposed_value, proposed_log_prob), lambda _: current_state, operand=None)
    return new_state
    
    
class RandomWalk(InferenceAlgorithm):
    def __init__(self, proposer: Callable[[jax.Array],dist.Distribution]) -> None:
        self.proposer = proposer
        self.jitted_kernel = False

    def make_kernel(self, gibbs_model: GibbsModel, step_number: int) -> Kernel:
        @jax.jit
        def _kernel(state: AbstractInferenceState, rng_key: PRNGKey) -> MHState:
            maybe_jit_warning(self, "jitted_kernel", "_rw_kernel", f"Inference step {step_number}: <RandomWalk at {hex(id(self))}>", to_shaped_arrays(state))
            X, Y = gibbs_model.split_trace(state.position)
            gibbs_model.set_Y(Y)
            log_prob = state.log_prob if hasattr(state, "log_prob") else gibbs_model.log_prob(X)
            current_mh_state = MHState(X, log_prob)
            next_mh_state = rw_kernel(rng_key, current_mh_state, gibbs_model.log_prob, self.proposer)
            next_mh_state = MHState(gibbs_model.combine_to_trace(next_mh_state.position, Y), next_mh_state.log_prob)
            return next_mh_state
        return _kernel
    
RW = RandomWalk

class InferenceRegime(ABC):
    @abstractmethod
    def __iter__(self) -> Generator["InferenceStep", Any, None]:
        pass

class InferenceStep(InferenceRegime):
    def __init__(self, variable_selector: VariableSelector, algo: InferenceAlgorithm) -> None:
        self.variable_selector = variable_selector
        self.algo = algo
    def __iter__(self):
        yield self

class Gibbs(InferenceRegime):
    def __init__(self, *subregimes: InferenceRegime) -> None:
        self.subregimes = subregimes
    def __iter__(self):
        for subregime in self.subregimes:
            if isinstance(subregime, InferenceStep):
                yield subregime
            else:
                assert isinstance(subregime, Gibbs)
                yield from subregime


def mcmc(slp: SLP, regime: InferenceRegime, n_samples: int, n_chains: int, rng_key: PRNGKey):
    kernels: List[Kernel] = []
    for step_number, inference_step in enumerate(regime):
        gibbs_model = GibbsModel(slp, inference_step.variable_selector)
        kernels.append(inference_step.algo.make_kernel(gibbs_model, step_number))

    @jax.jit
    def one_step(state: AbstractInferenceState, rng_key: PRNGKey) -> Tuple[AbstractInferenceState,Trace]:
        maybe_jit_warning(None, "", "_mcmc_step", slp.short_repr(), to_shaped_arrays(state))
        for kernel in kernels:
            rng_key, kernel_key = jax.random.split(rng_key)
            state = kernel(state, kernel_key)
        state = InferenceStateFromSubclass(state)
        return state, state.position
    
    keys = jax.random.split(rng_key, n_samples)

    _, result = jax.lax.scan(one_step, InferenceState(slp.decision_representative), keys)
    return result


