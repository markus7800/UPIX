import jax
import jax.numpy as jnp
from typing import Dict, Optional, List, Callable, Any, NamedTuple, Generic, TypeVar, Tuple, cast
from dccxjax.core import SLP, Model, sample_from_prior, slp_from_decision_representative
from ..types import Trace, PRNGKey, FloatArray, IntArray, StackedTrace, StackedTraces, StackedSampleValues, _unstack_sample_data
from dataclasses import dataclass
from .mcmc import MCMCRegime, MCMC, MCMCState, summarise_mcmc_info
from .estimate_Z import estimate_log_Z_for_SLP_from_prior
from time import time
from copy import deepcopy
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from ..utils import broadcast_jaxtree
from functools import reduce

class InferenceResult(ABC):
    @abstractmethod
    def concatentate(self, other: "InferenceResult") -> "InferenceResult":
        # fold left
        raise NotImplementedError

@dataclass
class MCMCInferenceResult(InferenceResult):
    value_tree: Any # either None or Pytree with leading axes (n_samples_per_chain, n_chains,...)
    last_state: MCMCState
    n_chains: int
    n_samples_per_chain: int
    def concatentate(self, other: InferenceResult) -> "MCMCInferenceResult":
        if not isinstance(other, MCMCInferenceResult):
            raise TypeError
        assert self.n_chains == other.n_chains
        if self.value_tree is None:
            assert other.value_tree is None
            value_tree = None
        else:
            assert other.value_tree is not None
            value_tree = jax.tree_map(lambda x, y: jax.lax.concatenate((x, y), 0), self.value_tree, other.value_tree)

        
        last_state = jax.tree_map(lambda x, y: jax.lax.concatenate((x, y), 0), self.value_tree, other.value_tree)
        return MCMCInferenceResult(value_tree, last_state, self.n_chains, self.n_samples_per_chain + other.n_samples_per_chain)
    
class LogWeightEstimate(ABC):
    @abstractmethod
    def combine_estimate(self, other: "LogWeightEstimate") -> "LogWeightEstimate":
        raise NotImplementedError
    

@dataclass
class LogWeightEstimateFromPrior(LogWeightEstimate):
    log_Z: FloatArray
    ESS: IntArray
    frac_in_support: FloatArray
    n_samples: int
    def combine_estimate(self, other: LogWeightEstimate) -> "LogWeightEstimateFromPrior":
        assert isinstance(other, LogWeightEstimateFromPrior)
        n_combined_samples = self.n_samples + other.n_samples
        a = self.n_samples / n_combined_samples
        
        log_Z = jax.numpy.logaddexp(self.log_Z + jax.lax.log(a), other.log_Z + jax.lax.log(1 - a))
        frac_in_support = self.frac_in_support * a + other.frac_in_support * (1 - a)

        return LogWeightEstimateFromPrior(log_Z, self.ESS + other.ESS, frac_in_support, n_combined_samples)


DCC_RESULT_TYPE = TypeVar("DCC_RESULT_TYPE")            

class AbstractDCC(ABC, Generic[DCC_RESULT_TYPE]):
    def __init__(self, model: Model, *ignore, verbose=0, **config_kwargs) -> None:
        if ignore:
            raise TypeError
        
        self.model = model
        self.config: Dict[str, Any] = config_kwargs

        self.verbose = verbose

    @abstractmethod
    def initialise_active_slps(self, active_slps: List[SLP], rng_key: PRNGKey):
        raise NotImplementedError
    
    @abstractmethod
    def run_inference(self, slp: SLP, rng_key: PRNGKey) -> InferenceResult:
        raise NotImplementedError
    
    @abstractmethod
    def estimate_log_weight(self, slp: SLP, rng_key: PRNGKey) -> LogWeightEstimate:
        raise NotImplementedError
    
    @abstractmethod
    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: PRNGKey):
        raise NotImplementedError
    
    @abstractmethod
    def combine_results(self, inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]]) -> DCC_RESULT_TYPE:
        pass


    def add_to_inference_results(self, slp: SLP, inference_result: InferenceResult):
        if slp not in self.inference_results:
            self.inference_results[slp] = []
        self.inference_results[slp].append(inference_result)

    def get_inference_results(self, slp: SLP):
        if slp not in self.inference_results:
            self.inference_results[slp] = []
        return self.inference_results[slp]

    def add_to_log_weight_estimates(self, slp: SLP, log_weight_estimate: LogWeightEstimate):
        if slp not in self.log_weight_estimates:
            self.log_weight_estimates[slp] = []
        self.log_weight_estimates[slp].append(log_weight_estimate)

    def get_log_weight_estimates(self, slp: SLP):
        if slp not in self.log_weight_estimates:
            self.log_weight_estimates[slp] = []
        return self.log_weight_estimates[slp]
        
    
    def run(self, rng_key: PRNGKey):
        # t0 = time()
        if self.verbose >= 2:
            tqdm.write("Start DCC:")

        self.active_slps: List[SLP] = []
        self.inactive_slps: List[SLP] = []

        self.inference_method_cache: Dict[SLP, Any] = dict()

        self.inference_results: Dict[SLP, List[InferenceResult]] = dict()
        self.log_weight_estimates: Dict[SLP, List[LogWeightEstimate]] = dict()        

        rng_key, init_key = jax.random.split(rng_key)
        self.initialise_active_slps(self.active_slps, init_key)

        self.iteration_counter = 0

        while len(self.active_slps) > 0:
            self.iteration_counter += 1
            if self.verbose >= 2:
                tqdm.write(f"Iteration {self.iteration_counter}:")

            for slp in self.active_slps:

                rng_key, slp_inference_key, slp_weight_estimate_key = jax.random.split(rng_key, 3)
                
                inference_result = self.run_inference(slp, slp_inference_key)
                self.add_to_inference_results(slp, inference_result)

                log_weight_estimate = self.estimate_log_weight(slp, slp_weight_estimate_key)
                self.add_to_log_weight_estimates(slp, log_weight_estimate)

            rng_key, update_key = jax.random.split(rng_key)
            self.update_active_slps(self.active_slps, self.inactive_slps, update_key)
        
        combined_result = self.combine_results(self.inference_results, self.log_weight_estimates)
        # t1 = time()
        # if self.verbose >= 2:
        #     tqdm.write(f"Finished in {t1-t0:.3f}s")
        return combined_result

DCC_COLLECT_TYPE = TypeVar("DCC_COLLECT_TYPE")
DCC_RESULT_QUERY_TYPE = TypeVar("DCC_RESULT_QUERY_TYPE")

@dataclass
class WeightedSample(Generic[DCC_COLLECT_TYPE]):
    values: StackedSampleValues[DCC_COLLECT_TYPE]
    log_weights: FloatArray
    def __repr__(self) -> str:
        return f"WeightedSample({self.values})"

@dataclass
class MCMCDCCResult(Generic[DCC_COLLECT_TYPE]):
    slp_log_weights: Dict[SLP, FloatArray]
    slp_weighted_samples: Dict[SLP, WeightedSample[DCC_COLLECT_TYPE]]

    def __repr__(self) -> str:
        return f"MCMC-DCCResult({len(self.slp_log_weights)} SLPs)"
    
    def pprint(self):
        log_Z_normaliser = self.get_log_weight_normaliser()
        print("MCMC-DCCResult {")
        for slp, log_weight in self.slp_log_weights.items():
            weighted_sample = self.slp_weighted_samples[slp]
            print(f"\t{slp.formatted()}: {weighted_sample.values} with prob={jnp.exp(log_weight - log_Z_normaliser).item():.6f}, log_Z={log_weight.item():6f}")
        print("}")

    def get_slps(self, predicate: Callable[[DCC_COLLECT_TYPE], bool] = lambda _: True) -> List[SLP]:
        return [slp for slp, weighted_sample in self.slp_weighted_samples.items() if predicate(weighted_sample.values.data)]
    
    # convience function for DCC_COLLECT_TYPE = Trace
    def get_slps_where_address_exists(self, address: str):
        return self.get_slps(lambda x: address in cast(Trace, x))
    
    # = model evidence if used DCC methods supports it, otherwise 0.
    def get_log_weight_normaliser(self):
        log_Zs = [log_Z for _, log_Z in self.slp_log_weights.items()]
        return jax.scipy.special.logsumexp(jnp.vstack(log_Zs))


    def _get_samples_for_slp(self, slp: SLP, unstack_chains: bool, mapper: Callable[[DCC_COLLECT_TYPE], DCC_RESULT_QUERY_TYPE]) -> Tuple[DCC_RESULT_QUERY_TYPE, FloatArray]:
        log_Z = self.slp_log_weights[slp]
        log_Z_normaliser = self.get_log_weight_normaliser()
        weighted_sample = self.slp_weighted_samples[slp]
        weights = jax.lax.exp(weighted_sample.log_weights - jax.scipy.special.logsumexp(weighted_sample.log_weights) + log_Z - log_Z_normaliser)

        assert weights.shape == (weighted_sample.values.N, weighted_sample.values.T)
        if unstack_chains:
            weights = _unstack_sample_data(weights) # has same shape as values (StackedSampleValues)
            values = weighted_sample.values.unstack().data
        else:
            values = weighted_sample.values.data

        return mapper(values), weights
    
    def get_samples_for_slp(self, slp: SLP, unstack_chains: bool = True, mapper: Callable[[DCC_COLLECT_TYPE], DCC_RESULT_QUERY_TYPE] = lambda x: x) -> Tuple[DCC_RESULT_QUERY_TYPE, FloatArray]:
        return self._get_samples_for_slp(slp, unstack_chains, mapper)
    
    # convience function for DCC_COLLECT_TYPE = Trace
    def get_samples_for_address_and_slp(self, address: str, slp: SLP, unstack_chains: bool = True):
        return self._get_samples_for_slp(slp, unstack_chains, lambda x: cast(Trace,x)[address])
    

    def _get_samples(self, unstack_chains: bool,
                     predicate: Callable[[DCC_COLLECT_TYPE], bool],
                     mapper: Callable[[DCC_COLLECT_TYPE], DCC_RESULT_QUERY_TYPE]
                     ) -> Tuple[Optional[DCC_RESULT_QUERY_TYPE], Optional[FloatArray], float]:
        
        undef_prob = 0.
        log_Z_normaliser = self.get_log_weight_normaliser()

        values: Optional[DCC_RESULT_QUERY_TYPE] = None
        weights: Optional[FloatArray] = None
        for slp, log_Z in self.slp_log_weights.items():
            weighted_sample = self.slp_weighted_samples[slp]
            slp_weights = jax.lax.exp(weighted_sample.log_weights - jax.scipy.special.logsumexp(weighted_sample.log_weights) + log_Z - log_Z_normaliser)
            assert slp_weights.shape == (weighted_sample.values.N, weighted_sample.values.T)

            if not predicate(weighted_sample.values.data):
                undef_prob += jax.lax.exp(log_Z).item()
            else:
                if unstack_chains:
                    slp_values = weighted_sample.values.unstack().data
                    slp_weights = _unstack_sample_data(slp_weights)
                else:
                    slp_values = weighted_sample.values.data
                slp_values = mapper(slp_values)

                if values is None:
                    values = slp_values
                    weights = slp_weights
                else:
                    values = jax.tree_map(lambda x, y: jax.lax.concatenate((x, y), 0), values, slp_values)
                    weights = jax.tree_map(lambda x, y: jax.lax.concatenate((x, y), 0), weights, slp_weights)


        return values, weights, undef_prob
                

    
    def get_samples(self, unstack_chains: bool = True,
                     predicate: Callable[[DCC_COLLECT_TYPE], bool] = lambda _: True,
                     mapper: Callable[[DCC_COLLECT_TYPE], DCC_RESULT_QUERY_TYPE] = lambda x: x
                     ):
        return self._get_samples(unstack_chains, predicate, mapper)


    # convience function for DCC_COLLECT_TYPE = Trace
    def get_samples_for_address(self, address: str, unstack_chains: bool = True):
        return self._get_samples(unstack_chains, lambda x: address in cast(Trace,x), lambda x: cast(Trace,x)[address])


class MCMCDCC(AbstractDCC[MCMCDCCResult[DCC_COLLECT_TYPE]], Generic[DCC_COLLECT_TYPE]):
    def __init__(self, model: Model, return_map: Callable[[Trace], DCC_COLLECT_TYPE] = lambda trace: trace, *ignore, verbose=0, **config_kwargs) -> None:
        super().__init__(model, verbose=verbose, *ignore, **config_kwargs)
        self.return_map = return_map

        self.init_n_samples: int = self.config.get("init_n_samples", 100)

        self.mcmc_collect_inference_info: bool = self.config.get("mcmc_collect_inference_info", True)
        self.mcmc_collect_for_all_traces: bool = self.config.get("mcmc_collect_for_all_traces", True)
        self.mcmc_optimise_memory_with_early_return_map: bool = self.config.get("mcmc_optimise_memory_with_early_return_map", False)
        self.mcmc_n_chains: int = self.config.get("mcmc_n_chains", 4)
        self.mcmc_n_samples_per_chain: int = self.config.get("mcmc_n_samples_per_chain", 1_000)

        self.estimate_weight_n_samples: int = self.config.get("estimate_weight_n_samples", 1_000_000)


    # should populate active_slps
    def initialise_active_slps(self, active_slps: List[SLP], rng_key: PRNGKey):
        if self.verbose >= 2:
            tqdm.write("Initialise active SLPS.")
        discovered_slps: List[SLP] = []

        # default samples from prior
        for _ in tqdm(range(self.init_n_samples), desc="Search SLPs from prior"):
            rng_key, key = jax.random.split(rng_key)
            trace = sample_from_prior(self.model, key)

            if all(slp.path_indicator(trace) == 0 for slp in discovered_slps):
                slp = slp_from_decision_representative(self.model, trace)
                if self.verbose >= 2:
                    tqdm.write(f"Discovered SLP {slp.formatted()}.")
                discovered_slps.append(slp)
        active_slps.extend(discovered_slps)

        # # select only a-priori likely paths
        # if self.min_prior_path_probability > 0.0:
        #     for slp in discovered_slps:
        #         rng_key, key = jax.random.split(rng_key)
        #         log_Z, ESS, frac_in_support = estimate_log_Z_for_SLP_from_prior(slp, self.estimate_weight_n_samples, rng_key)
        #         if frac_in_support > self.min_prior_path_probability:
        #             tqdm.write(f"Make SLP {slp.formatted()} active (frac_in_support={frac_in_support.item():.4f}).")
        #             self.add_to_log_weight_estimates(slp, LogWeightEstimateFromPrior(log_Z, ESS, frac_in_support, self.estimate_weight_n_samples))
        #             active_slps.append(slp)
        # else:
        #     tqdm.write(f"Make all discovered SLPs active.")
        #     active_slps.extend(discovered_slps)

    @abstractmethod
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        raise NotImplementedError

    def get_MCMC(self, slp: SLP) -> MCMC:
        if slp in self.inference_method_cache:
            mcmc = self.inference_method_cache[slp]
            assert isinstance(mcmc, MCMC)
            return mcmc
        regime = self.get_MCMC_inference_regime(slp)
        def mcmc_return_map(state: MCMCState):
            if self.mcmc_collect_for_all_traces:
                if self.mcmc_optimise_memory_with_early_return_map:
                    return self.return_map(state.position)
                else:
                    return (state.position, state.log_prob)
            else:
                return None
        mcmc = MCMC(slp, regime, self.mcmc_n_chains,
                    collect_inference_info=self.mcmc_collect_inference_info,
                    return_map=mcmc_return_map,
                    progress_bar=self.verbose >= 1)
        self.inference_method_cache[slp] = mcmc
        return mcmc


    def run_inference(self, slp: SLP, rng_key: PRNGKey) -> InferenceResult:
        mcmc = self.get_MCMC(slp)
        inference_results = self.inference_results.get(slp, [])
        if len(inference_results) > 0:
            last_result = inference_results[-1]
            assert isinstance(last_result, MCMCInferenceResult)
            init_positions = StackedTrace(last_result.last_state.position, mcmc.n_chains)
            log_prob = last_result.last_state.log_prob
        else:
            init_positions = StackedTrace(broadcast_jaxtree(slp.decision_representative, (mcmc.n_chains,)), mcmc.n_chains)
            log_prob = broadcast_jaxtree(slp.log_prob(slp.decision_representative), (mcmc.n_chains,))
        
        last_state, return_result = mcmc.run(rng_key, init_positions, log_prob, n_samples_per_chain=self.mcmc_n_samples_per_chain)
        if self.verbose >= 2 and self.mcmc_collect_inference_info:
            assert last_state.infos is not None
            info_str = "MCMC Infos:"
            for step, info in enumerate(last_state.infos):
                info_str += f"\n\t Step {step}: {summarise_mcmc_info(info, self.mcmc_n_samples_per_chain)}"
            tqdm.write(info_str)
        
        # TODO: only run MCMC on gpu and handle all result data on CPU
        cpu = jax.devices("cpu")[0]
        return_result = jax.device_put(return_result, cpu)

        return MCMCInferenceResult(return_result, last_state, mcmc.n_chains, self.mcmc_n_samples_per_chain)
    
    def estimate_log_weight(self, slp: SLP, rng_key: PRNGKey) -> LogWeightEstimate:
        log_Z, ESS, frac_in_support = estimate_log_Z_for_SLP_from_prior(slp, self.estimate_weight_n_samples, rng_key)
        if self.verbose >= 2:
            tqdm.write(f"Estimated log weight for {slp.formatted()}: {log_Z.item()} (ESS={ESS.item():,.0f})")
        return LogWeightEstimateFromPrior(log_Z, ESS, frac_in_support, self.estimate_weight_n_samples)

    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: PRNGKey):
        # TODO: replace default with LMH
        inactive_slps.extend(active_slps)
        active_slps.clear()

    
    def compute_slp_log_weight(self, log_weight_estimates: Dict[SLP, LogWeightEstimate]) -> Dict[SLP, FloatArray]:
        log_Zs: List[FloatArray] = []
        for estimate in log_weight_estimates.values():
            assert isinstance(estimate, LogWeightEstimateFromPrior)
            log_Zs.append(estimate.log_Z)

        # log_Z_normaliser = jax.scipy.special.logsumexp(jnp.vstack(log_Zs))
        slp_log_weights: Dict[SLP, FloatArray] = {}
        for slp, estimate in log_weight_estimates.items():
            assert isinstance(estimate, LogWeightEstimateFromPrior)
            slp_log_weights[slp] = estimate.log_Z

        return slp_log_weights

    def get_slp_weighted_samples(self, inference_results: Dict[SLP, InferenceResult]) -> Dict[SLP, WeightedSample[DCC_COLLECT_TYPE]]:
        slp_weighted_samples: Dict[SLP, WeightedSample[DCC_COLLECT_TYPE]] = {}
        for slp, inference_result in inference_results.items():
            assert isinstance(inference_result, MCMCInferenceResult)
            if self.mcmc_collect_for_all_traces:
                if self.mcmc_optimise_memory_with_early_return_map:
                    # assert isinstance(inference_result.value_tree, DCC_COLLECT_TYPE)
                    values: DCC_COLLECT_TYPE = inference_result.value_tree
                else:
                    # assert isinstance(inference_result.value_tree, Trace)
                    values = self.return_map(inference_result.value_tree[0])

                weighted_samples = WeightedSample(
                    StackedSampleValues(values, inference_result.n_samples_per_chain, inference_result.n_chains),
                    jnp.zeros((inference_result.n_samples_per_chain, inference_result.n_chains), float)
                )
            
            else:
                # number of samples per chain = number of times MCMC was performed for SLP
                # because we only stored last state
                assert inference_result.value_tree is None
                n_mcmc = inference_result.last_state.iteration.size
                n_chains = inference_result.n_chains
                values = jax.tree_map(lambda v: v.reshape((n_mcmc,n_chains) + v.shape[1:]), inference_result.last_state.position)
                weighted_samples = WeightedSample(
                    StackedSampleValues(values, n_mcmc, n_chains),
                    jnp.zeros((n_mcmc,n_chains), float)
                )

            slp_weighted_samples[slp] = weighted_samples


        return slp_weighted_samples

    def combine_results(self, inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]]) -> MCMCDCCResult[DCC_COLLECT_TYPE]:
        combined_inference_results: Dict[SLP, InferenceResult] = {slp: reduce(lambda x, y: x.concatentate(y), results) for slp, results in inference_results.items()}
        combined_log_weight_estimates: Dict[SLP, LogWeightEstimate] = {slp: reduce(lambda x, y: x.combine_estimate(y), results) for slp, results in log_weight_estimates.items()}

        slp_log_weights = self.compute_slp_log_weight(combined_log_weight_estimates)
        slp_weighted_samples = self.get_slp_weighted_samples(combined_inference_results)

        return MCMCDCCResult(slp_log_weights, slp_weighted_samples)
    