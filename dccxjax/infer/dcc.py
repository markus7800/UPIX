import jax
import jax.numpy as jnp
from typing import Dict, Optional, List, Callable, Any, NamedTuple, Generic, TypeVar, Tuple, cast
from dccxjax.core import SLP, Model, sample_from_prior, slp_from_decision_representative
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
    def combine(self, other: "InferenceResult") -> "InferenceResult":
        # fold left
        raise NotImplementedError

class LogWeightEstimate(ABC):
    @abstractmethod
    def combine(self, other: "LogWeightEstimate") -> "LogWeightEstimate":
        raise NotImplementedError
    
    
class BaseDCCResult:
    slp_log_weights: Dict[SLP, FloatArray]
    
    def get_slp_weights(self, predicate: Callable[[SLP], bool] = lambda _: True) -> Dict[SLP, float]:
        log_Z_normaliser = self.get_log_weight_normaliser()
        slp_weights = {slp: jnp.exp(log_weight - log_Z_normaliser).item() for slp, log_weight in self.slp_log_weights.items()}
        normaliser = sum(slp_weights.values())
        # numerical inprecision may lead to weigths being not normalised because we normalised in "log" space
        slp_weights = {slp: weight/normaliser for slp, weight in slp_weights.items() if predicate(slp)}
        return slp_weights
    def get_slps(self, predicate: Callable[[SLP], bool] = lambda _: True) -> List[SLP]:
        return [slp for slp in self.slp_log_weights.keys() if predicate(slp)]
    
    def get_slp(self, predicate: Callable[[SLP], bool]) -> Optional[SLP]:
        slps = self.get_slps(predicate)
        if len(slps) == 0:
            return None
        elif len(slps) == 1:
            return slps[0]
        else:
            print("Warn: multiple slps for predicate")
            return slps[0]

    # convenience function for DCC_COLLECT_TYPE = Trace
    def get_slps_where_address_exists(self, address: str):
        return self.get_slps(lambda slp: address in slp.decision_representative)
    
    # = model evidence if used DCC methods supports it, otherwise 0.
    def get_log_weight_normaliser(self):
        log_Zs = [log_Z for _, log_Z in self.slp_log_weights.items()]
        return jax.scipy.special.logsumexp(jnp.vstack(log_Zs))
    
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

def initialise_active_slps_from_prior(model: Model, verbose: int, init_n_samples: int, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: PRNGKey):
    if verbose >= 2:
        tqdm.write("Initialise active SLPS.")
    discovered_slps: List[SLP] = []

    # default samples from prior
    for _ in tqdm(range(init_n_samples), desc="Search SLPs from prior"):
        rng_key, key = jax.random.split(rng_key)
        trace = sample_from_prior(model, key)

        if model.equivalence_map is not None:
            trace = model.equivalence_map(trace)

        if all(slp.path_indicator(trace) == 0 for slp in discovered_slps):
            slp = slp_from_decision_representative(model, trace)
            if verbose >= 2:
                tqdm.write(f"Discovered SLP {slp.formatted()}.")
            discovered_slps.append(slp)

    active_slps.extend(discovered_slps)