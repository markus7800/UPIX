import jax
import jax.numpy as jnp
from typing import Dict, Optional, List, Callable, Any, NamedTuple, Generic, TypeVar, Tuple, cast
from dccxjax.core import SLP, Model
from dccxjax.types import Trace, PRNGKey, FloatArray, IntArray, StackedTrace, StackedTraces, StackedSampleValues, _unstack_sample_data
from dccxjax.infer.dcc.abstract_dcc import JaxTask, InferenceTask, EstimateLogWeightTask, InferenceResult, LogWeightEstimate, AbstractDCC, BaseDCCResult, initialise_active_slps_from_prior, ParallelisationType, is_sequential
from dataclasses import dataclass
from dccxjax.infer.variational_inference.vi import Guide, ADVI, ADVIState, Optimizer
from dccxjax.infer.variational_inference.optimizers import Adagrad
from abc import abstractmethod
__all__ = [
    "VIDCC",
    "ADVIInferenceResult",
    "LogWeightEstimateFromADVI",
]

@jax.tree_util.register_dataclass
@dataclass
class ADVIInferenceResult(InferenceResult):
    last_state: ADVIState
    def combine_results(self, other: InferenceResult) -> InferenceResult:
        # not used in default implementation of VIDCC
        assert isinstance(other, ADVIInferenceResult)
        # take advi with most steps
        if self.last_state.iteration < other.last_state.iteration:
            return other
        else:
            return self


@jax.tree_util.register_dataclass
@dataclass
class LogWeightEstimateFromADVI(LogWeightEstimate):
    log_Z: FloatArray
    n_samples: int
    def combine_estimates(self, other: LogWeightEstimate) -> "LogWeightEstimateFromADVI":
        # not used in default implementation of VIDCC
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
    def pprint(self, *, sortkey: str = "logweight"):
        log_Z_normaliser = self.get_log_weight_normaliser()
        print("VI-DCCResult {")
        slp_log_weights_list = self.get_log_weights_sorted(sortkey)
        for slp, log_weight in slp_log_weights_list:
            guide = self.slp_guides[slp]
            print(f"\t{slp.formatted()}: with prob={jnp.exp(log_weight - log_Z_normaliser).item():.6f}, log_Z={log_weight.item():6f}")
        print("}")


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
        initialise_active_slps_from_prior(self.model, self.verbose, self.init_n_samples, active_slps, inactive_slps, rng_key)
    
    @abstractmethod
    def get_guide(self, slp: SLP) -> Guide:
        raise NotImplementedError
    
    def get_ADVI(self, slp: SLP) -> ADVI:
        if slp in self.inference_method_cache:
            advi = self.inference_method_cache[slp]
            assert isinstance(advi, ADVI)
            return advi
        guide = self.get_guide(slp)
        advi = ADVI(slp, guide, self.advi_optimizer, self.advi_L, pconfig=self.pconfig,
                    show_progress=self.verbose >= 1 and is_sequential(self.pconfig),
                    shared_progressbar=self.shared_progress_bar)
        self.inference_method_cache[slp] = advi
        return advi
    
    # def estimate_log_weight(self, slp: SLP, rng_key: PRNGKey) -> LogWeightEstimate:
    #     inference_results = self.inference_results.get(slp, [])
    #     if len(inference_results) > 0:
    #         last_result = inference_results[-1]
    #         assert isinstance(last_result, ADVIInferenceResult)
    #         guide = self.get_ADVI(slp).get_updated_guide(last_result.last_state)
    #         Xs, lqs = guide.sample_and_log_prob(rng_key, shape=(self.elbo_estimate_n_samples,))
    #         lps = jax.vmap(slp.log_prob)(Xs)
    #         elbo = jnp.mean(lps - lqs)
    #         return LogWeightEstimateFromADVI(elbo, self.elbo_estimate_n_samples)
    #     else:
    #         raise Exception("In VIDCC we should perform one run of ADVI before estimate_log_weight to estimate elbo from guide")
        
    def make_estimate_log_weight_task(self, slp: SLP, rng_key: PRNGKey) -> EstimateLogWeightTask:
        inference_results = self.inference_results.get(slp, [])
        if len(inference_results) > 0:
            last_result = inference_results[-1]
            assert isinstance(last_result, ADVIInferenceResult)
            guide = self.get_ADVI(slp).get_updated_guide(last_result.last_state)
            def _task(rng_key: PRNGKey):
                Xs, lqs = guide.sample_and_log_prob(rng_key, shape=(self.elbo_estimate_n_samples,))
                lps = jax.vmap(slp.log_prob)(Xs)
                elbo = jnp.mean(lps - lqs)
                return LogWeightEstimateFromADVI(elbo, self.elbo_estimate_n_samples)
            return EstimateLogWeightTask(_task, (rng_key,))
        else:
            raise Exception("In VIDCC we should perform one run of ADVI before estimate_log_weight to estimate elbo from guide")

    
    def make_inference_task(self, slp: SLP, rng_key: PRNGKey) -> InferenceTask:
        advi = self.get_ADVI(slp)
        inference_results = self.inference_results.get(slp, [])
        
        def _f_run_pre_info():
            # if self.parallelisation.type != ParallelisationType.Sequential:
            #     return f"Start ADVI for {slp.formatted()}"
            return ""
        def _f_run_post_info(result: InferenceResult):
            assert isinstance(result, ADVIInferenceResult)
            if self.share_progress_bar:
                return f"Finished ADVI for {slp.formatted()}"
            else:
                return ""
            
        if len(inference_results) > 0:
            last_result = inference_results[-1]
            assert isinstance(last_result, ADVIInferenceResult)
            # sets iteration count = 0 (may affect optimizers schedule)
            # iteration is also used in progressbar (so we would have to add additional counter if we want to set iteration to different start value)
            def _task_continue(rng_key, last_state):
                last_state, elbo = advi.continue_run(rng_key, last_state, n_iter=self.advi_n_iter, iteration=jnp.array(0,int))
                return ADVIInferenceResult(last_state)
            return InferenceTask(_task_continue, (rng_key, last_result.last_state), _f_run_pre_info, _f_run_post_info)
        else:
            def _task_run(rng_key):
                last_state, elbo = advi.run(rng_key, n_iter=self.advi_n_iter)
                return ADVIInferenceResult(last_state)
            return InferenceTask(_task_run, (rng_key,), _f_run_pre_info, _f_run_post_info)
        
        
    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        inactive_slps.extend(active_slps)
        active_slps.clear()

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
            advi = self.get_ADVI(slp)
            guide = advi.get_updated_guide(inference_result.last_state)
            slp_guides[slp] = guide
        return slp_guides

    def combine_results(self, inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]]) -> VIDCCResult:
        # combined_inference_results: Dict[SLP, InferenceResult] = {slp: reduce(lambda x, y: x.combine(y), results) for slp, results in inference_results.items()}
        # combined_log_weight_estimates: Dict[SLP, LogWeightEstimate] = {slp: reduce(lambda x, y: x.combine(y), results) for slp, results in log_weight_estimates.items()}

        # take latest result
        combined_inference_results: Dict[SLP, InferenceResult] = {slp: results[-1] for slp, results in inference_results.items()}
        combined_log_weight_estimates: Dict[SLP, LogWeightEstimate] = {slp: results[-1] for slp, results in log_weight_estimates.items()}

        slp_log_weights = self.compute_slp_log_weight(combined_log_weight_estimates)
        slp_guides = self.get_slp_guides(combined_inference_results)

        return VIDCCResult(slp_log_weights, slp_guides)
    