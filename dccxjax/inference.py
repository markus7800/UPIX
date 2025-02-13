from .slp_gen import Model, SLP, sample_from_prior, slp_from_decision_representative, estimate_Z_for_SLP_from_mcmc, estimate_Z_for_SLP_from_prior
from .slp_gen import decision_representative_from_partial_trace
from .variable_selector import VariableSelector
from .gibbs import GibbsModel
from typing import List, Dict, Optional, Tuple, Generator, Any, NamedTuple, Callable, Set
import jax
from abc import ABC, abstractmethod
from .types import PRNGKey, Trace
import numpyro.distributions as dist
from .utils import maybe_jit_warning, to_shaped_arrays, logger
from dataclasses import dataclass
from time import time

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
            kernel_keys = jax.random.split(kernel_key, n_chains)
            state = jax.vmap(kernel)(kernel_keys, state)
        state = InferenceState(state.position)
        return state, state.position if collect_states else None
    
    return one_step


def get_initial_inference_state(slp: SLP, n_chains: int):
    # add leading dimension by broadcasting, i.e. X of shape (m,n,...) has now shape (n_chains,m,n,...)
    # and X[i,m,n,...] = X[j,m,n,...] for all i, j
    return InferenceState(jax.tree_map(lambda x: jax.lax.broadcast(x, (n_chains,)), slp.decision_representative))

    
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


def _unstack_chains(values: jax.Array):
    shape = values.shape
    assert len(shape) >= 2
    var_dim = () if len(shape) < 3 else (shape[2],)
    n_samples = shape[0]
    n_chains = shape[1]
    return jax.lax.reshape(values, (n_samples * n_chains, *var_dim))

def unstack_chains(Xs: Trace):
    return jax.tree_map(_unstack_chains, Xs)

def n_samples(Xs: Trace):
    _, some_entry = next(iter(Xs.items()))
    return some_entry.shape[0] * some_entry.shape[1]


class DCC_Result:
    def __init__(self, intermediate_states: bool) -> None:
        self.intermediate_states = intermediate_states
        self.samples: Dict[SLP, Trace] = dict()
        # samples have shape (n_samples, n_chain, dim(var))
        self.Zs: Dict[SLP, jax.Array] = dict()
    def add_samples(self, slp: SLP, samples: Trace, Z: Optional[jax.Array]):

        #  shape of trace entry for var:
        #  (n_samples_per_chain, n_chain, dim(var)) if multi_chain=True and intermediate_states=True
        #  (n_chain, dim(var)) if multi_chain=True and intermediate_states=False
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

    def get_samples_for_address_and_slp(self, address: str, slp: SLP, unstack_chains: bool = True):
        Z_slp = self.Zs[slp]
        Z = sum(z for _, z in self.Zs.items())
        samples = self.samples[slp]
        assert address in samples
        samples_for_address = samples[address]
        weigths_for_address = jax.lax.full_like(samples_for_address, Z_slp / Z)

        if unstack_chains:
            samples_for_address = _unstack_chains(samples_for_address)
            weigths_for_address = _unstack_chains(weigths_for_address)

        return samples_for_address, weigths_for_address

    def get_samples_for_address(self, address: str, unstack_chains: bool = True):
        undef_prob = 0.
        Z = sum(z for _, z in self.Zs.items())
        samples_for_address: Optional[jax.Array] = None
        weigths_for_address: Optional[jax.Array] = None
        for slp, samples in self.samples.items():
            Z_slp = self.Zs[slp]
            if address not in samples:
                undef_prob += Z_slp / Z
            else:
                weights = jax.lax.full_like(samples[address], Z_slp / Z)
                if samples_for_address is None:
                    samples_for_address = samples[address]
                    weigths_for_address = weights
                else:
                    assert weigths_for_address is not None
                    samples_for_address = jax.lax.concatenate((samples_for_address, samples[address]), 0)
                    weigths_for_address = jax.lax.concatenate((weigths_for_address, weights), 0)

        if unstack_chains and samples_for_address is not None and weigths_for_address is not None:
            samples_for_address = _unstack_chains(samples_for_address)
            weigths_for_address = _unstack_chains(weigths_for_address)
        
        return samples_for_address, weigths_for_address, undef_prob
    
    def n_samples(self):
        n = 0
        for _, samples in self.samples.items():
            n += n_samples(samples)
        return n


from copy import deepcopy

def propose_new_slps_from_last_positions(slp: SLP, last_positions: Trace, active_slps: List[SLP], proposed_slps: Dict[SLP,int], n_chains: int, rng_key: PRNGKey, scale: float):
    if len(slp.branching_variables) == 0:
        return
    
    # we want to find a jittered_position such that decision_representative_from_partial_trace(jittered_position) is not in support of any SLP
    # in decision_representative_from_partial_trace we run the model and take values from jittered_position where possible else sample from prior
    # in this process we must encounter at least one decision (provided that len(slp.branching_variables) > 0)
    # furthermore all SLPs share at least one (this first encountered) decision
    # thus SLPs share at least one branching variable
    # so we randomly perturb the values in last_positions for all branching_variables (= jittered_position)
    # the addresses of dr := decision_representative_from_partial_trace(jittered_position) may be different from the addresses of jittered_position / last_positions
    # for each other_slp how can we test if dr is in the support by probing with jittered_position?
    # dr and jittered_position have the same path in the model until they disagree on one decision
    # up to this decision they sample the same addresses (a subset of the addresses of jittered_position)
    # It may happen that due to the random pertubation, dr and jittered_position have the same path until a sample statement is encountered where its address is missing in jittered_position
    # in this case, dr may or may not be in the support of other_slp


    # shape of values is (n_chains, val(dims))
    jittered_positions: Trace = dict()
    for addr, values in last_positions.items():
        if addr in slp.branching_variables:
            rng_key, sample_key = jax.random.split(rng_key)
            jittered_positions[addr] = values + scale * jax.random.normal(sample_key, values.shape)
        else:
            jittered_positions[addr] = values

    for i in range(n_chains):
        partial_X: Trace = {addr: values[i,:] if len(values.shape) == 2 else values[i] for addr, values in jittered_positions.items()}
        rng_key, sample_key = jax.random.split(rng_key)
        decision_representative = decision_representative_from_partial_trace(slp.model, partial_X, sample_key)

        in_support_of_any_slp = False
        for other_slp in active_slps:
            if other_slp.path_indicator(decision_representative):
                in_support_of_any_slp = True
                break
        for other_slp, count in proposed_slps.items():
            if other_slp.path_indicator(decision_representative):
                in_support_of_any_slp = True
                proposed_slps[other_slp] = count + 1
        
        if not in_support_of_any_slp:
            new_slp = slp_from_decision_representative(slp.model, decision_representative)
            proposed_slps[new_slp] = 1
            print(f"Proposed new slp {new_slp.short_repr()}", new_slp.formatted())


    # def in_support_of_other_slp(other_slp: SLP):
    #     assert len(other_slp.branching_variables.intersection(slp.branching_variables)) > 0
    #     extended_jittered_positions: Trace = dict()
    #     for addr in other_slp.decision_representative.keys():
    #         if addr in jittered_positions:
    #             extended_jittered_positions[addr] = jittered_positions[addr]
    #         else:
    #             extended_jittered_positions[addr] = jax.lax.broadcast(other_slp.decision_representative[addr], (n_chains,))
    #     return jax.vmap(other_slp._path_indicator)(extended_jittered_positions)
    
    # in_support_of_any_slp = jax.lax.full((n_chains,), 0)
    # for other_slp in active_slps:
    #     if other_slp.decision_representative.keys() <= jittered_positions.keys():
    #         # if we change values in trace, then some sample statements may not be encountered anymore
    #         # to avoid adding the same SLP multiple times we check for subset instead of equality
    #         in_support_of_any_slp = in_support_of_any_slp | jax.vmap(other_slp._path_indicator)(jittered_positions)
    #         if in_support_of_any_slp.sum() == n_chains:
    #             break

    # for other_slp, count in proposed_slps.items():
    #     # if other_slp.decision_representative.keys() <= jittered_positions.keys():
    #     #     in_support_of_proposed_slp = jax.vmap(other_slp._path_indicator)(jittered_positions)
    #     #     in_support_of_any_slp = in_support_of_any_slp | in_support_of_proposed_slp
    #     #     # proposed_slps[other_slp] = count + in_support_of_any_slp.sum().item()
    #     # else:
    #     extended_jittered_positions: Trace = dict()
    #     for addr in other_slp.branching_variables:
    #         if addr in jittered_positions:
    #             extended_jittered_positions[addr] = jittered_positions[addr]
    #         else:
    #             extended_jittered_positions[addr] = jax.lax.broadcast(other_slp.decision_representative[addr], (n_chains,))
    #     in_support_of_proposed_slp = jax.vmap(other_slp._path_indicator)(extended_jittered_positions)
    #     in_support_of_any_slp = in_support_of_any_slp | in_support_of_proposed_slp

    # print(in_support_of_any_slp)
            
    # for i in range(n_chains):
    #     if in_support_of_any_slp[i] == 0:
    #         partial_X: Trace = {addr: values[i,:] if len(values.shape) == 2 else values[i] for addr, values in jittered_positions.items()}
    #         decision_representative = decision_representative_from_partial_trace(slp.model, partial_X)
    #         new_slp = slp_from_decision_representative(slp.model, decision_representative)

    #         extended_jittered_positions: Trace = dict()
    #         for addr in new_slp.branching_variables:
    #             if addr in jittered_positions:
    #                 extended_jittered_positions[addr] = jittered_positions[addr]
    #             else:
    #                 extended_jittered_positions[addr] = jax.lax.broadcast(new_slp.decision_representative[addr], (n_chains,))
    #         in_support_of_proposed_slp = jax.vmap(new_slp._path_indicator)(extended_jittered_positions)
    #         in_support_of_any_slp = in_support_of_any_slp | in_support_of_proposed_slp

    #         proposed_slps[new_slp] = 0
    #         print(f"Proposed new slp {new_slp.short_repr()}", new_slp.branching_decisions.to_human_readable().splitlines()[-1])
    



def dcc(model: Model, regime: InferenceRegime, rng_key: PRNGKey, config: DCC_Config):
    all_slps: List[SLP] = []
    active_slps: List[SLP] = []
    proposed_slps: Dict[SLP,int] = dict()
    slp_to_mcmc_step = dict()
    slp_to_number_of_mcmc_runs: Dict[SLP,int] = dict()

    for _ in range(config.n_samples_from_prior):
        rng_key, key = jax.random.split(rng_key)
        X = sample_from_prior(model, key)
        slp = slp_from_decision_representative(model, X)

        if all(slp.path_indicator(X) == 0 for slp in active_slps):
            active_slps.append(slp)
            slp_to_mcmc_step[slp] = get_inference_regime_mcmc_step_for_slp(slp, deepcopy(regime), config.n_chains, config.collect_intermediate_chain_states)

    combined_result = DCC_Result(intermediate_states=config.collect_intermediate_chain_states)
    
    t0 = time()
    while True:
        did_mcmc = False
        for slp in active_slps:
            if slp_to_number_of_mcmc_runs.get(slp, 0) > 0:
                continue
            slp_to_number_of_mcmc_runs[slp] = slp_to_number_of_mcmc_runs.get(slp, 0) + 1
            did_mcmc = True

            mcmc_step = slp_to_mcmc_step[slp]
            print(f"Run mcmc for {slp.short_repr()}", slp.formatted())
            
            init = get_initial_inference_state(slp, config.n_chains)
            rng_key, key1, key2, key3 = jax.random.split(rng_key, 4)

            mcmc_keys = jax.random.split(key1, config.n_samples_per_chain)
            last_state, all_positions = jax.lax.scan(mcmc_step, init, mcmc_keys)
            last_positions = last_state.position
            # shape of entries of all_positions is (n_samples_per_chain, n_chains, dim(var))
            # shape of entries of last_positions is (n_chains, dim(var))

            Z = estimate_Z_for_SLP_from_prior(slp, config.n_samples_for_Z_est, key2)
            combined_result.add_samples(slp, all_positions if config.collect_intermediate_chain_states else last_positions, Z)

            # Z2 = estimate_Z_for_SLP_from_mcmc(slp, 1.0, config.n_samples_for_Z_est // (config.n_samples_per_chain * config.n_chains) + 1, key2, unstack_chains(all_positions))
            # print(f"{Z=} vs {Z2=}")

            propose_new_slps_from_last_positions(slp, last_positions, active_slps, proposed_slps, config.n_chains, key3, 1.)
            add_to_active: List[SLP] = []
            for proposed_slp, count in proposed_slps.items():
                if count > 10:
                    add_to_active.append(proposed_slp)
                    print(f"Add to active: slp {proposed_slp.short_repr()}", proposed_slp.formatted())
            for proposed_slp in add_to_active:
                active_slps.append(proposed_slp)
                slp_to_mcmc_step[proposed_slp] = get_inference_regime_mcmc_step_for_slp(proposed_slp, deepcopy(regime), config.n_chains, config.collect_intermediate_chain_states)
                del proposed_slps[proposed_slp]


        if not did_mcmc:
            break

    t1 = time()
    t = t1 - t0
    n_samples = combined_result.n_samples()
    print("DCC: collected", n_samples, f"samples in {t:.3f}s ({n_samples / t / 1000:.3f}K/s)")

    # slp_desc = [slp.branching_decisions.to_human_readable().splitlines()[-1] for slp in active_slps]
    # for desc in sorted(slp_desc):
    #     print(desc)


    return combined_result