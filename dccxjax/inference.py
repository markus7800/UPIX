from .slp_gen import SLP, sample_from_prior, slp_from_X, estimate_Z_for_SLP
from .variable_selector import VariableSelector
from .gibbs import GibbsModel
from typing import List, Dict, Optional, Tuple, Generator, Any, NamedTuple, Callable
import jax
from .samplecontext import Model
from abc import ABC, abstractmethod
from .types import PRNGKey, Trace
import numpyro.distributions as dist
from .utils import maybe_jit_warning, to_shaped_arrays
from dataclasses import dataclass


@jax.tree_util.register_dataclass
@dataclass
class InferenceState(object):
    position: Trace
    
    
Kernel = Callable[[PRNGKey,InferenceState],InferenceState]

class InferenceAlgorithm(ABC):
    @abstractmethod
    def make_kernel(self, gibbs_model: GibbsModel, step_number: int) -> Kernel:
        raise NotImplementedError

@jax.tree_util.register_dataclass
@dataclass
class MHState(InferenceState):
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

#         def _kernel(rng_key: PRNGKey, X: InferenceState) -> MHState:
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
        def _rw_kernel(rng_key: PRNGKey, state: InferenceState) -> MHState:
            maybe_jit_warning(self, "jitted_kernel", "_rw_kernel", f"Inference step {step_number}: <RandomWalk at {hex(id(self))}>", to_shaped_arrays(state))
            X, Y = gibbs_model.split_trace(state.position)
            gibbs_model.set_Y(Y)
            log_prob = getattr(state, "log_prob") if hasattr(state, "log_prob") else gibbs_model.log_prob(X)
            current_mh_state = MHState(X, log_prob)
            next_mh_state = rw_kernel(rng_key, current_mh_state, gibbs_model.log_prob, self.proposer)
            next_mh_state = MHState(gibbs_model.combine_to_trace(next_mh_state.position, Y), next_mh_state.log_prob)
            return next_mh_state
        return _rw_kernel
    
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


# TODO: we could always vmap even if n_chains = 1 to simplify
def get_inference_regime_mcmc_step_for_slp(slp: SLP, regime: InferenceRegime, n_chains: int, collect_states: bool):
    kernels: List[Kernel] = []
    for step_number, inference_step in enumerate(regime):
        gibbs_model = GibbsModel(slp, inference_step.variable_selector)
        kernels.append(inference_step.algo.make_kernel(gibbs_model, step_number))

    @jax.jit
    def one_step(state: InferenceState, rng_key: PRNGKey) -> Tuple[InferenceState,Optional[Trace]]:
        maybe_jit_warning(None, "", "_mcmc_step", slp.short_repr(), to_shaped_arrays(state))
        for kernel in kernels:
            rng_key, kernel_key = jax.random.split(rng_key)
            if n_chains > 1:
                kernel_keys = jax.random.split(kernel_key, n_chains)
                state = jax.vmap(kernel)(kernel_keys, state)
            else:
                state = kernel(kernel_key, state)
        state = InferenceState(state.position)
        return state, state.position if collect_states else None
    
    return one_step


def get_initial_inference_state(slp: SLP, n_chains: int):
    if n_chains > 1:
        # add leading dimension by broadcasting, i.e. X of shape (m,n,...) has now shape (n_chains,m,n,...)
        # and X[i,m,n,...] = X[j,m,n,...] for all i, j
        return InferenceState(jax.tree_map(lambda x: jax.lax.broadcast(x, (n_chains,)), slp.decision_representative))
    else:
        return InferenceState(slp.decision_representative)
    
def mcmc(slp: SLP, regime: InferenceRegime, n_samples: int, n_chains: int, rng_key: PRNGKey, collect_states: bool = True):
    
    keys = jax.random.split(rng_key, n_samples)

    mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, regime, n_chains, collect_states)

    init = get_initial_inference_state(slp, n_chains)

    last_state, all_positions = jax.lax.scan(mcmc_step, init, keys)
 
    return all_positions if collect_states else last_state.position


@dataclass
class DCC_Config:
    n_samples_from_prior: int
    n_chains: int
    collect_intermediate_chain_states: bool
    n_samples_per_chain: int
    n_samples_for_Z_est: int


class DCC_Result:
    def __init__(self, multi_chain: bool, intermediate_states: bool) -> None:
        self.multi_chain = multi_chain
        self.intermediate_states = intermediate_states
        self.samples: Dict[SLP, Trace] = dict()
        # samples have shape (n_samples, n_chain, dim(var)) if multi_chain else (n_samples, dim(var))
        self.Zs: Dict[SLP, jax.Array] = dict()
    def add_samples(self, slp: SLP, samples: Trace, Z: Optional[jax.Array]):

        print("add samples for", slp)
        for addr, values in samples.items():
            print(addr, values.shape)

        #  shape of trace entry for var:
        #  (n_samples_per_chain, n_chain, dim(var)) if multi_chain=True and intermediate_states=True
        #  (n_chain, dim(var)) if multi_chain=True and intermediate_states=False
        #  (n_samples_per_chain, dim(var)) if multi_chain=False and intermediate_states=True
        #  (dim(var),) if multi_chain=False and intermediate_states=False
        # if dim(var) == 1 dimension of trace entry is ommited, e.g. (n_samples_per_chain, n_chain) in the first case and () in the last case
        if not self.intermediate_states:
            samples = jax.tree_map(lambda x: jax.lax.broadcast(x, (1,)), samples)
        if slp not in self.samples:
            self.samples[slp] = samples
            assert Z is not None
            self.Zs[slp] = Z
        else:
            # samples have shape (n_samples_per_chain, n_chain, dim(var)) or (n_samples_per_chain, dim(var)) where n_samples_per_chain can be 1
            prev_samples = self.samples[slp]
            self.samples[slp] = jax.tree_map(lambda x, y: jax.lax.concatenate((x, y), 0), prev_samples, samples)
            if Z is not None:
                self.Zs[slp] = Z

        print("now have")
        for addr, values in self.samples[slp].items():
            print(addr, values.shape)

from copy import deepcopy

def dcc(model: Model, regime: InferenceRegime, rng_key: PRNGKey, config: DCC_Config):

    active_slps: List[SLP] = []
    slp_to_mcmc_step = dict()

    for _ in range(config.n_samples_from_prior):
        rng_key, key = jax.random.split(rng_key)
        X = sample_from_prior(model, key)
        slp = slp_from_X(model, X)

        if all(slp.path_indicator(X) == 0 for slp in active_slps):
            active_slps.append(slp)
            slp_to_mcmc_step[slp] = get_inference_regime_mcmc_step_for_slp(slp, deepcopy(regime), config.n_chains, config.collect_intermediate_chain_states)

    combined_result = DCC_Result(multi_chain=config.n_chains > 1, intermediate_states=config.collect_intermediate_chain_states)
    
    for slp in active_slps:
        mcmc_step = slp_to_mcmc_step[slp]
        init = get_initial_inference_state(slp, config.n_chains)
        rng_key, key1, key2 = jax.random.split(rng_key, 3)

        mcmc_keys = jax.random.split(key1, config.n_samples_per_chain)
        last_state, all_positions = jax.lax.scan(mcmc_step, init, mcmc_keys)
        last_position = last_state.position

        Z_est_keys = jax.random.split(key2, config.n_samples_for_Z_est)
        Z = estimate_Z_for_SLP(slp, Z_est_keys)
        combined_result.add_samples(slp, all_positions if config.collect_intermediate_chain_states else last_position, Z)


    return combined_result