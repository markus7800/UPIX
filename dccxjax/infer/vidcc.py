import jax
import jax.numpy as jnp
from typing import Dict, Optional, List, Callable, Any, NamedTuple, Generic, TypeVar, Tuple, cast
from dccxjax.core import SLP, Model, sample_from_prior, slp_from_decision_representative
from ..types import Trace, PRNGKey, FloatArray, IntArray, StackedTrace, StackedTraces, StackedSampleValues, _unstack_sample_data
from .dcc import InferenceResult, LogWeightEstimate, AbstractDCC, BaseDCCResult
from dataclasses import dataclass
from .vi import Guide, ADVI, Adagrad, ADVIState, Optimizer
from tqdm.auto import tqdm
from abc import abstractmethod
from functools import reduce

__all__ = [
    "VIDCC",
]

@dataclass
class ADVIInferenceResult(InferenceResult):
    last_state: ADVIState
    def combine(self, other: InferenceResult) -> InferenceResult:
        assert isinstance(other, ADVIInferenceResult)
        # take advi with most steps
        if self.last_state.iteration < other.last_state.iteration:
            return other
        else:
            return self


@dataclass
class LogWeightEstimateFromADVI(LogWeightEstimate):
    log_Z: FloatArray
    n_samples: int
    def combine(self, other: LogWeightEstimate) -> "LogWeightEstimateFromADVI":
        assert isinstance(other, LogWeightEstimateFromADVI)
        n_combined_samples = self.n_samples + other.n_samples
        a = self.n_samples / n_combined_samples
        
        log_Z = jax.numpy.logaddexp(self.log_Z + jax.lax.log(a), other.log_Z + jax.lax.log(1 - a))

        return LogWeightEstimateFromADVI(log_Z, n_combined_samples)
    
    def get_estimate(self):
        return self.log_Z
    
@dataclass
class VIDCCResult(BaseDCCResult):
    slp_log_weights: Dict[SLP, FloatArray]
    slp_guides: Dict[SLP, Guide]

    def __repr__(self) -> str:
        return f"VI-DCCResult({len(self.slp_log_weights)} SLPs)"
    
    def pprint(self):
        pass


    # TODO
    
class VIDCC(AbstractDCC[VIDCCResult]):
    def __init__(self, model: Model, *ignore, verbose=0, **config_kwargs) -> None:
        super().__init__(model, verbose=verbose, *ignore, **config_kwargs)

        self.init_n_samples: int = self.config.get("init_n_samples", 100)

        self.advi_n_iter: int = self.config.get("advi_n_iter", 1000)
        self.advi_L: int = self.config.get("advi_L", 1)
        self.advi_optimizer: Optimizer = self.config.get("advi_optimizer", Adagrad(1.))

        self.elbo_estimate_n_samples: int = self.config.get("elbo_estimate_n_samples", 100)

    # should populate active_slps
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: PRNGKey):
        if self.verbose >= 2:
            tqdm.write("Initialise active SLPS.")
        discovered_slps: List[SLP] = []

        # default samples from prior
        for _ in tqdm(range(self.init_n_samples), desc="Search SLPs from prior"):
            rng_key, key = jax.random.split(rng_key)
            trace = sample_from_prior(self.model, key)

            if self.model.equivalence_map is not None:
                trace = self.model.equivalence_map(trace)

            if all(slp.path_indicator(trace) == 0 for slp in discovered_slps):
                slp = slp_from_decision_representative(self.model, trace)
                if self.verbose >= 2:
                    tqdm.write(f"Discovered SLP {slp.formatted()}.")
                discovered_slps.append(slp)

        active_slps.extend(discovered_slps)
    

    @abstractmethod
    def get_guide(self, slp: SLP) -> Guide:
        raise NotImplementedError
    
    def get_ADVI(self, slp: SLP) -> ADVI:
        if slp in self.inference_method_cache:
            mcmc = self.inference_method_cache[slp]
            assert isinstance(mcmc, ADVI)
            return mcmc
        guide = self.get_guide(slp)
        mcmc = ADVI(slp, guide, self.advi_optimizer, 1, progress_bar=self.verbose >= 1)
        self.inference_method_cache[slp] = mcmc
        return mcmc
    
    def estimate_log_weight(self, slp: SLP, rng_key: PRNGKey) -> LogWeightEstimate:
        inference_results = self.inference_results.get(slp, [])
        if len(inference_results) > 0:
            last_result = inference_results[-1]
            assert isinstance(last_result, ADVIInferenceResult)
            guide = self.get_ADVI(slp).guide
            last_optimizer_state = last_result.last_state.optimizer_state
            guide.update_params(self.advi_optimizer.get_params_fn(last_optimizer_state))

            Xs, lqs = jax.vmap(guide.sample_and_log_prob)(jax.random.split(rng_key, self.elbo_estimate_n_samples))
            lps = jax.vmap(slp.log_prob)(Xs)
            elbo = jnp.mean(lps - lqs)
            return LogWeightEstimateFromADVI(elbo, self.elbo_estimate_n_samples)

        else:
            raise Exception("In VIDCC we should perform one run of ADVI before estimate_log_weight to estimate elbo from guide")


    def run_inference(self, slp: SLP, rng_key: PRNGKey) -> InferenceResult:
        advi = self.get_ADVI(slp)
        inference_results = self.inference_results.get(slp, [])
        if len(inference_results) > 0:
            last_result = inference_results[-1]
            assert isinstance(last_result, ADVIInferenceResult)
            # starts iterations again from 0 by default (may affect optimizers, see Adam) TODO: change this?
            last_state, elbo = advi.continue_run(rng_key, last_result.last_state, n_iter=self.advi_n_iter)
        else:
            last_state, elbo = advi.run(rng_key, n_iter=self.advi_n_iter)
        if self.verbose >= 2:
            # TODO: report some stats
            pass
        return ADVIInferenceResult(last_state)
        
    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        inactive_slps.extend(active_slps)
        active_slps.clear()
        # successive halfing as default in subclass

    def compute_slp_log_weight(self, log_weight_estimates: Dict[SLP, LogWeightEstimate]) -> Dict[SLP, FloatArray]:
        slp_log_weights: Dict[SLP, FloatArray] = {}
        for slp, estimate in log_weight_estimates.items():
            assert isinstance(estimate, LogWeightEstimateFromADVI)
            slp_log_weights[slp] = estimate.get_estimate()
        return slp_log_weights

    def get_slp_guides(self, inference_results: Dict[SLP, InferenceResult]) -> Dict[SLP, Guide]:
        slp_guides: Dict[SLP, Guide] = {}
        for slp, inference_result in inference_results.items():
            assert isinstance(inference_result, ADVIInferenceResult)
            last_optimizer_state = inference_result.last_state.optimizer_state
            guide = self.get_ADVI(slp).guide
            params = self.advi_optimizer.get_params_fn(last_optimizer_state)
            guide.update_params(params)
            slp_guides[slp] = guide
        return slp_guides

    def combine_results(self, inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]]) -> VIDCCResult:
        combined_inference_results: Dict[SLP, InferenceResult] = {slp: reduce(lambda x, y: x.combine(y), results) for slp, results in inference_results.items()}
        combined_log_weight_estimates: Dict[SLP, LogWeightEstimate] = {slp: reduce(lambda x, y: x.combine(y), results) for slp, results in log_weight_estimates.items()}

        slp_log_weights = self.compute_slp_log_weight(combined_log_weight_estimates)
        slp_guides = self.get_slp_guides(combined_inference_results)

        return VIDCCResult(slp_log_weights, slp_guides)
    
