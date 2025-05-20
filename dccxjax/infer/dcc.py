import jax
import jax.numpy as jnp
from typing import Dict, Optional, List, Callable, Any, NamedTuple, Generic, TypeVar, Tuple, cast
from dccxjax.core import SLP, Model
from ..types import Trace, PRNGKey, FloatArray, IntArray
from dataclasses import dataclass
from .mcmc import MCMCState
from tqdm.auto import tqdm
from abc import ABC, abstractmethod

__all__ = [
    "InferenceResult",
    "LogWeightEstimate",

]

class InferenceResult(ABC):
    @abstractmethod
    def concatentate(self, other: "InferenceResult") -> "InferenceResult":
        # fold left
        raise NotImplementedError

class LogWeightEstimate(ABC):
    @abstractmethod
    def combine_estimate(self, other: "LogWeightEstimate") -> "LogWeightEstimate":
        raise NotImplementedError
    

DCC_RESULT_TYPE = TypeVar("DCC_RESULT_TYPE")            

class AbstractDCC(ABC, Generic[DCC_RESULT_TYPE]):
    def __init__(self, model: Model, *ignore, verbose=0, **config_kwargs) -> None:
        if ignore:
            raise TypeError
        
        self.model = model
        self.config: Dict[str, Any] = config_kwargs

        self.verbose = verbose

    @abstractmethod
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: PRNGKey):
        raise NotImplementedError
    
    @abstractmethod
    def run_inference(self, slp: SLP, rng_key: PRNGKey) -> InferenceResult:
        raise NotImplementedError
    
    @abstractmethod
    def estimate_log_weight(self, slp: SLP, rng_key: PRNGKey) -> LogWeightEstimate:
        raise NotImplementedError
    
    @abstractmethod
    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
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
        self.initialise_active_slps(self.active_slps, self.inactive_slps, init_key)

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
            self.update_active_slps(self.active_slps, self.inactive_slps, self.inference_results, self.log_weight_estimates, update_key)
        
        combined_result = self.combine_results(self.inference_results, self.log_weight_estimates)
        # t1 = time()
        # if self.verbose >= 2:
        #     tqdm.write(f"Finished in {t1-t0:.3f}s")
        return combined_result

