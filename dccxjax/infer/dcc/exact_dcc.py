from abc import abstractmethod
from typing import List, Dict, Optional
from dccxjax.types import FloatArray, IntArray
from dccxjax.core import Model, SLP
from dccxjax.types import PRNGKey
from dccxjax.infer.exact import Factor, compute_factors, get_greedy_elimination_order, variable_elimination, get_supports
from dccxjax.infer.dcc.abstract_dcc import InferenceTask, EstimateLogWeightTask, InferenceResult, LogWeightEstimate, AbstractDCC, BaseDCCResult, initialise_active_slps_from_prior
from tqdm.auto import tqdm
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import time

__all__ = [
    "ExactDCC"
]


@jax.tree_util.register_dataclass
@dataclass
class ExactInferenceResult(InferenceResult,LogWeightEstimate):
    factor: Factor
    log_evidence: FloatArray
    def combine_results(self, other: InferenceResult) -> InferenceResult:
        print("Warning: Tried to combine exact results.")
        return self
    def combine_estimates(self, other: LogWeightEstimate) -> LogWeightEstimate:
        print("Warning: Tried to combine exact estimates.")
        return self


@dataclass
class ExactDCCResult(BaseDCCResult):
    slp_log_weights: Dict[SLP, FloatArray]
    slp_to_factor: Dict[SLP, Factor]

    def __repr__(self) -> str:
        return f"Exact-DCCResult({len(self.slp_log_weights)} SLPs)"
    
    def pprint(self, *, sortkey: str = "logweight"):
        log_Z_normaliser = self.get_log_weight_normaliser()
        slp_log_weights_list = self.get_log_weights_sorted(sortkey)
        print("Exact-DCCResult {")
        for slp, log_weight in slp_log_weights_list:
            factor = self.slp_to_factor[slp]
            print(f"\t{slp.formatted()}: {factor} with prob={jnp.exp(log_weight - log_Z_normaliser).item():.6f}, log_Z={log_weight.item():6f}")
        print("}")
        
    # TODO
    
class ExactDCC(AbstractDCC[ExactDCCResult]):
    def __init__(self, model: Model, *ignore, verbose=0, **config_kwargs) -> None:
        super().__init__(model, *ignore, verbose=verbose, **config_kwargs)
        
        self.init_n_samples: int = self.config.get("init_n_samples", 100)
        self.jit_inference: bool = self.config.get("jit_inference", False)
        
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: PRNGKey):
        initialise_active_slps_from_prior(self.model, self.verbose, self.init_n_samples, active_slps, inactive_slps, rng_key)

    @abstractmethod
    def get_query_variables(self, slp: SLP) -> List[str]:
        raise NotImplementedError
    
    def make_estimate_log_weight_task(self, slp: SLP, rng_key: PRNGKey) -> EstimateLogWeightTask:
        inference_results = self.inference_results.get(slp, [])
        if len(inference_results) > 0:
            result = inference_results[0]
            assert isinstance(result, ExactInferenceResult)
            return EstimateLogWeightTask(lambda: result, ())
        else:
            raise Exception("In SMCDCC we should perform one run of SMC before estimate_log_weight to reuse estimate")
    
    def get_factors(self, slp: SLP, supports: Dict[str, Optional[IntArray]]) -> List[Factor]:
        return compute_factors(slp, supports, True)
    
    def get_elimination_order(self, slp: SLP, factors: List[Factor]) -> List[str]:
        t0 = time.time()
        marginal_variables = self.get_query_variables(slp)
        elimination_order = get_greedy_elimination_order(factors, marginal_variables)
        t1 = time.time()
        if self.verbose >= 2:
            tqdm.write(f"Computed elimination order in {t1-t0:.3f}s")
        return elimination_order
    
    def make_inference_task(self, slp: SLP, rng_key: PRNGKey) -> InferenceTask:
        inference_results = self.inference_results.get(slp, [])
        if len(inference_results) > 0:
            def _return_last():
                return inference_results[0]
            return InferenceTask(_return_last, ())
        else:            
            t0 = time.time()
            supports = get_supports(slp)
            t1 = time.time()
            if self.verbose >= 2:
                tqdm.write(f"Computed supports in {t1-t0:.3f}s")
                
            def _compute_exact(supports):
                t0 = time.time()
                factors = self.get_factors(slp, supports)
                t1 = time.time()
                if self.verbose >= 2 and not self.jit_inference:
                    tqdm.write(f"Computed factors in {t1-t0:.3f}s")
                elimination_order = self.get_elimination_order(slp, factors)
                t0 = time.time()
                @jax.jit
                def _ve_jitted(factors: List[Factor]):
                    return variable_elimination(factors, elimination_order)
                result_factor, log_evidence = _ve_jitted(factors)
                t1 = time.time()
                if self.verbose >= 2 and not self.jit_inference:
                    tqdm.write(f"Performed variable_elimination in {t1-t0:.3f}s")
                    tqdm.write(f"Log-evidence: {log_evidence.item():.6f}")
                
                return ExactInferenceResult(result_factor, log_evidence)
            return InferenceTask(jax.jit(_compute_exact) if self.jit_inference else _compute_exact, (supports, ))
                
    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: FloatArray):
        inactive_slps.extend(active_slps)
        active_slps.clear()

    def combine_results(self, inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]]) -> ExactDCCResult:
        # take latest result
        slp_to_factor: Dict[SLP, Factor] = dict()
        slp_log_weights: Dict[SLP, FloatArray] = dict()
        for slp, results in inference_results.items():
            result = results[-1]
            assert isinstance(result, ExactInferenceResult)
            slp_to_factor[slp] = result.factor
            slp_log_weights[slp] = result.log_evidence

        return ExactDCCResult(slp_log_weights, slp_to_factor)