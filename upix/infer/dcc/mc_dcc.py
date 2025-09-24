import jax
import jax.numpy as jnp
from typing import Dict, Optional, List, Callable, Any, NamedTuple, Generic, TypeVar, Tuple, cast
from upix.core import SLP, Model, slp_from_decision_representative
from upix.types import Trace, PRNGKey, FloatArray, IntArray, StackedTrace, StackedTraces, StackedSampleValues, _unstack_sample_data
from dataclasses import dataclass
from time import time
from copy import deepcopy
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from functools import reduce
from upix.infer.mcmc.lmh_global import lmh
from upix.infer.variable_selector import AllVariables, VariableSelector
from upix.infer.dcc.abstract_dcc import InferenceResult, LogWeightEstimate, AbstractDCC, BaseDCCResult, initialise_active_slps_from_prior

__all__ = [
    "MCDCC",
    "DCC_COLLECT_TYPE",
    "T",
]

T = TypeVar("T")

DCC_COLLECT_TYPE = TypeVar("DCC_COLLECT_TYPE")
DCC_RESULT_QUERY_TYPE = TypeVar("DCC_RESULT_QUERY_TYPE")

class BaseWeightedSample(Generic[DCC_COLLECT_TYPE]):
    values: StackedSampleValues[DCC_COLLECT_TYPE]
    def n_chains(self): return self.values.n_chains()
    def n_samples_per_chain(self): return self.values.n_samples_per_chain()
    def n_samples(self): return self.values.n_samples()

@dataclass
class LogWeightedSample(BaseWeightedSample, Generic[DCC_COLLECT_TYPE]):
    values: StackedSampleValues[DCC_COLLECT_TYPE]
    log_weights: FloatArray
    def __repr__(self) -> str:
        return f"LogWeightedSample({self.values})"
    def exp(self) -> "WeightedSample[DCC_COLLECT_TYPE]":
        return WeightedSample(self.values, jax.lax.exp(self.log_weights))

@dataclass
class WeightedSample(BaseWeightedSample, Generic[DCC_COLLECT_TYPE]):
    values: StackedSampleValues[DCC_COLLECT_TYPE]
    weights: FloatArray
    def __repr__(self) -> str:
        return f"WeightedSample({self.values})"
    def log(self) -> LogWeightedSample[DCC_COLLECT_TYPE]:
        return LogWeightedSample(self.values, jax.lax.log(self.weights))
    
class MCInferenceResult(InferenceResult, ABC, Generic[DCC_COLLECT_TYPE]):
    @abstractmethod
    def get_weighted_sample(self, return_map: Callable[[Trace],DCC_COLLECT_TYPE]) -> LogWeightedSample[DCC_COLLECT_TYPE]:
        raise NotImplementedError
    
class MCLogWeightEstimate(LogWeightEstimate, ABC):
    @abstractmethod
    def get_estimate(self):
        raise NotImplementedError

@dataclass
class MCDCCResult(BaseDCCResult, Generic[DCC_COLLECT_TYPE]):
    slp_log_weights: Dict[SLP, FloatArray]
    slp_weighted_samples: Dict[SLP, LogWeightedSample[DCC_COLLECT_TYPE]]

    def __repr__(self) -> str:
        return f"MC-DCCResult({len(self.slp_log_weights)} SLPs)"

    def sprint(self, *, sortkey: str = "logweight"):
        s = "MC-DCCResult {\n"
        if len(self.slp_log_weights) > 0:
            log_Z_normaliser = self.get_log_weight_normaliser()
            slp_log_weights_list = self.get_log_weights_sorted(sortkey)
            for slp, log_weight in slp_log_weights_list:
                weighted_sample = self.slp_weighted_samples[slp]
                s += f"\t{slp.formatted()}: {weighted_sample.values} with prob={jnp.exp(log_weight - log_Z_normaliser).item():.6f}, log_Z={log_weight.item():6f}\n"
        s += "}\n"
        return s
    
    def pprint(self, *, sortkey: str = "logweight"):
        print(self.sprint(sortkey=sortkey))

    def _get_samples_for_slp(self, slp: SLP,
                             mapper: Callable[[DCC_COLLECT_TYPE], DCC_RESULT_QUERY_TYPE],
                             selector: Callable[[jax.Array], jax.Array],
                             ) -> StackedSampleValues[Tuple[DCC_RESULT_QUERY_TYPE, FloatArray]]:
        log_Z = self.slp_log_weights[slp]
        log_Z_normaliser = self.get_log_weight_normaliser()
        weighted_sample = self.slp_weighted_samples[slp]
        weights = jax.lax.exp(weighted_sample.log_weights - jax.scipy.special.logsumexp(weighted_sample.log_weights) + log_Z - log_Z_normaliser)

        assert weights.shape == (weighted_sample.values.N, weighted_sample.values.T)

        values = mapper(weighted_sample.values.data)
        
        weights = selector(weights)
        values = jax.tree.map(selector,values)

        weights = weights / weights.sum() # selector results in unnormalised weights
        
        return StackedSampleValues((values, weights), weighted_sample.values.N, weighted_sample.values.T)
    
    def get_samples_for_slp(self, slp: SLP,
                             mapper: Callable[[DCC_COLLECT_TYPE], DCC_RESULT_QUERY_TYPE] = lambda x: x,
                             selector: Callable[[jax.Array], jax.Array] = lambda w: w
                             ) -> StackedSampleValues[Tuple[DCC_RESULT_QUERY_TYPE, FloatArray]]:
        return self._get_samples_for_slp(slp, mapper, selector)
    
    # convenience function for DCC_COLLECT_TYPE = Trace
    def get_samples_for_address_and_slp(self, address: str, slp: SLP, sample_ixs:Any=slice(None,None), chain_ixs:Any=slice(None,None)):
        return self._get_samples_for_slp(slp, 
            lambda x: cast(Trace,x)[address][sample_ixs,chain_ixs,...],
            lambda x: x[sample_ixs,chain_ixs]
        )

    def _get_samples(self,
                     predicate: Callable[[DCC_COLLECT_TYPE], bool],
                     mapper: Callable[[DCC_COLLECT_TYPE], DCC_RESULT_QUERY_TYPE],
                     selector: Callable[[jax.Array], jax.Array]
                     ) -> Tuple[Optional[StackedSampleValues[Tuple[DCC_RESULT_QUERY_TYPE,FloatArray]]], float]:
        
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
                slp_values = weighted_sample.values.data
                slp_values = mapper(slp_values)

                slp_weights = selector(slp_weights)
                slp_values = jax.tree.map(selector,slp_values)

                if values is None:
                    values = slp_values
                    weights = slp_weights
                else:
                    values = jax.tree.map(lambda x, y: jax.lax.concatenate((x, y), 0), values, slp_values)
                    weights = jax.tree.map(lambda x, y: jax.lax.concatenate((x, y), 0), weights, slp_weights)

        if values is not None:
            assert weights is not None
            weights = weights / weights.sum() # selector results in unnormalised weights
            return StackedSampleValues((values, weights), *weights.shape), undef_prob
        else:
            return None, undef_prob
                

    def get_samples(self,
                     predicate: Callable[[DCC_COLLECT_TYPE], bool] = lambda _: True,
                     mapper: Callable[[DCC_COLLECT_TYPE], DCC_RESULT_QUERY_TYPE] = lambda x: x,
                     selector: Callable[[jax.Array], jax.Array] = lambda x: x,
                     ):
        return self._get_samples(predicate, mapper, selector)
        

    # convenience function for DCC_COLLECT_TYPE = Trace
    def get_samples_for_address(self, address: str, sample_ixs:Any=slice(None,None), chain_ixs:Any=slice(None,None)):
        return self._get_samples(
            lambda x: address in cast(Trace,x),
            lambda x: (cast(Trace,x)[address]),
            lambda x: x[sample_ixs,chain_ixs]
        )
    
class MCDCC(AbstractDCC[MCDCCResult[DCC_COLLECT_TYPE]]):
    def __init__(self, model: Model, return_map: Callable[[Trace], DCC_COLLECT_TYPE] = lambda trace: trace, *ignore, verbose=0, **config_kwargs) -> None:
        super().__init__(model, verbose=verbose, *ignore, **config_kwargs)
        self.return_map = return_map

        self.init_n_samples: int = self.config.get("init_n_samples", 100)

        self.estimate_weight_n_samples: int = self.config.get("estimate_weight_n_samples", 1_000_000)

        self.n_lmh_update_samples: int = self.config.get("n_lmh_update_samples", 1_000)
        self.lmh_variable_selector: VariableSelector = self.config.get("lmh_variable_selector", AllVariables())
        self.one_inference_run_per_slp: bool = self.config.get("one_inference_run_per_slp", True)

        self.max_active_slps: int = self.config.get("max_active_slps", 10)
        self.max_new_active_slps: int = self.config.get("max_new_active_slps", 10)
        self.max_iterations: int = self.config.get("max_iterations", 10)

    # should populate active_slps
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: PRNGKey):
        initialise_active_slps_from_prior(self.model, self.verbose, self.init_n_samples, active_slps, inactive_slps, rng_key, self.disable_progress)
    

    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        inactive_slps.extend(active_slps)
        active_slps.clear()

        if self.iteration_counter == self.max_iterations:
            return

        combined_inference_results: Dict[SLP, InferenceResult] = {slp: reduce(lambda x, y: x.combine_results(y), results) for slp, results in inference_results.items()}
        combined_log_weight_estimates: Dict[SLP, LogWeightEstimate] = {slp: reduce(lambda x, y: x.combine_estimates(y), results) for slp, results in log_weight_estimates.items()}

        self.combined_inference_results = {slp: [combined_result] for slp, combined_result in combined_inference_results.items()} # small optimisation to avoid repeated combination
        self.log_weight_estimates = {slp: [combined_estimate] for slp, combined_estimate in combined_log_weight_estimates.items()} # small optimisation to avoid repeated combination

        slp_log_weights = self.compute_slp_log_weight(combined_log_weight_estimates)
        slps: List[SLP] = []
        log_weight_list: List[FloatArray] = []
        for slp, log_weight in slp_log_weights.items():
            slps.append(slp)
            log_weight_list.append(log_weight)
        log_weights = jnp.array(log_weight_list)
        
        slp_to_proposal_prob: Dict[SLP, FloatArray] = dict()
        for _ in tqdm(range(self.n_lmh_update_samples), desc="Determining new active SLPs with LMH"):
            rng_key, select_slp_key, select_trace_key, lmh_key = jax.random.split(rng_key, 4)
            # select SLP proportional to the logweight
            slp = slps[jax.random.categorical(select_slp_key, log_weights)]
            slp_results = combined_inference_results[slp]
            assert isinstance(slp_results, MCInferenceResult)
            
            # select trace from SLP inference result
            weighted_sample: LogWeightedSample[Trace] = slp_results.get_weighted_sample(lambda x: x)
            trace_ix = jax.random.categorical(select_trace_key, weighted_sample.log_weights.reshape(-1))
            sample_ix, chain_ix = jnp.unravel_index(trace_ix, weighted_sample.log_weights.shape)
            trace: Trace = jax.tree.map(lambda v: v[sample_ix, chain_ix, ...], weighted_sample.values.data)

            # propose new trace with lmh
            trace_proposed, acceptance_log_prob = lmh(self.model, self.lmh_variable_selector, trace, lmh_key)

            if self.model.equivalence_map is not None:
                trace_proposed = self.model.equivalence_map(trace_proposed)

            # check if we know slp of proposed trace
            matched_slp = next(filter(lambda _slp: _slp.path_indicator(trace_proposed) != 0, inactive_slps), None)
            if matched_slp is None:
                matched_slp = slp_from_decision_representative(self.model, trace_proposed)
                if self.verbose >= 2:
                    tqdm.write(f"Discovered SLP {matched_slp.formatted()}.")
                inactive_slps.append(matched_slp)

            slp_to_proposal_prob[matched_slp] = jnp.logaddexp(slp_to_proposal_prob.get(matched_slp, -jnp.inf), acceptance_log_prob)

        # pick top with respect to acceptance probability
        slp_to_proposal_prob_list = list(slp_to_proposal_prob.items())
        slp_to_proposal_prob_list.sort(key=lambda v: v[1].item(), reverse=True)

        new_active_slp_count = 0
        for slp, prob in slp_to_proposal_prob_list:
            if len(active_slps) >= self.max_active_slps:
                break

            already_performed_inference = slp in slp_log_weights
            if self.one_inference_run_per_slp and already_performed_inference:
                continue
            if (not already_performed_inference) and (new_active_slp_count >= self.max_new_active_slps):
                continue

            tqdm.write(f"Make SLP {slp.formatted()} active (already performed inference = {already_performed_inference}).")
            active_slps.append(slp)
            inactive_slps.remove(slp)
            new_active_slp_count += (not already_performed_inference)
            

    def compute_slp_log_weight(self, log_weight_estimates: Dict[SLP, LogWeightEstimate]) -> Dict[SLP, FloatArray]:
        slp_log_weights: Dict[SLP, FloatArray] = {}
        for slp, estimate in log_weight_estimates.items():
            assert isinstance(estimate, MCLogWeightEstimate)
            slp_log_weights[slp] = estimate.get_estimate()

        return slp_log_weights

    def get_slp_weighted_samples(self, inference_results: Dict[SLP, InferenceResult]) -> Dict[SLP, LogWeightedSample[DCC_COLLECT_TYPE]]:
        slp_weighted_samples: Dict[SLP, LogWeightedSample[DCC_COLLECT_TYPE]] = {}
        for slp, inference_result in inference_results.items():
            assert isinstance(inference_result, MCInferenceResult)
            slp_weighted_samples[slp] = inference_result.get_weighted_sample(self.return_map)

        return slp_weighted_samples

    def combine_results(self, inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]]) -> MCDCCResult[DCC_COLLECT_TYPE]:
        combined_inference_results: Dict[SLP, InferenceResult] = {slp: reduce(lambda x, y: x.combine_results(y), results) for slp, results in inference_results.items()}
        combined_log_weight_estimates: Dict[SLP, LogWeightEstimate] = {slp: reduce(lambda x, y: x.combine_estimates(y), results) for slp, results in log_weight_estimates.items()}

        slp_log_weights = self.compute_slp_log_weight(combined_log_weight_estimates)
        slp_weighted_samples = self.get_slp_weighted_samples(combined_inference_results)

        return MCDCCResult(slp_log_weights, slp_weighted_samples)
    
