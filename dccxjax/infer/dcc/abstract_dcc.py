import jax
import jax.numpy as jnp
from typing import Dict, Optional, List, Callable, Any, NamedTuple, Generic, TypeVar, Tuple, cast
from dccxjax.core import SLP, Model, sample_from_prior, slp_from_decision_representative
from dccxjax.types import Trace, PRNGKey, FloatArray, IntArray
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dccxjax.infer.dcc.dcc_types import InferenceResult, InferenceTask, LogWeightEstimate
import os
import threading
from queue import Queue, ShutDown
from dccxjax.infer.dcc.cpu_multiprocess import process_worker, _export_flat
from jax.tree_util import tree_flatten, tree_unflatten
import jax.export

__all__ = [

]

class BaseDCCResult:
    slp_log_weights: Dict[SLP, FloatArray]
    
    def get_log_weights_sorted(self, sortkey: str = "logweight"):
        assert sortkey in ("logweight", "slp")
        slp_log_weights_list = list(self.slp_log_weights.items())
        if sortkey == "logweight":
            slp_log_weights_list.sort(key = lambda v: v[1].item())
        else:
            slp_log_weights_list.sort(key = lambda v: v[0].sort_key())
        return slp_log_weights_list
    
    def pprint(self, *, sortkey: str = "logweight"):
        assert sortkey in ("log_weight", "slp")
        log_Z_normaliser = self.get_log_weight_normaliser()
        print("BaseDCCResult {")
        slp_log_weights_list = self.get_log_weights_sorted(sortkey)
        for slp, log_weight in slp_log_weights_list:
            print(f"\t{slp.formatted()}: with prob={jnp.exp(log_weight - log_Z_normaliser).item():.6f}, log_Z={log_weight.item():6f}")
        print("}")
    
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

        self.parallelisation: str = self.config.get("parallelisation", "none")
        self.num_processes: int = self.config.get("num_processes", os.cpu_count())
        self.pin_cpus: bool = self.config.get("pin_cpus", False)

    @abstractmethod
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: PRNGKey):
        raise NotImplementedError
    
    @abstractmethod
    def make_inference_task(self, slp: SLP, rng_key: PRNGKey) -> InferenceTask:
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
        assert self.parallelisation in ("none", "multi-processing")
        # t0 = time()
        if self.verbose >= 2:
            tqdm.write("Start DCC:")

        self.active_slps: List[SLP] = []
        self.inactive_slps: List[SLP] = []

        task_queue = Queue()
        result_queue = Queue()
        threads: List[threading.Thread] = []
        if self.parallelisation == "multi-processing":
            for i in range(self.num_processes):
                t = threading.Thread(target=process_worker, args=(task_queue, result_queue, i, i if self.pin_cpus else None), daemon=True)
                t.start()
                threads.append(t)


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

            if self.parallelisation == "none":
                for slp in tqdm(self.active_slps, total=len(self.active_slps), desc=f"Iteration {self.iteration_counter}", position=0):
                    rng_key, slp_inference_key, slp_weight_estimate_key = jax.random.split(rng_key, 3)
                    
                    inference_task = self.make_inference_task(slp, slp_inference_key)
                    inference_result = inference_task.run()
                    self.add_to_inference_results(slp, inference_result)

                    log_weight_estimate = self.estimate_log_weight(slp, slp_weight_estimate_key)
                    self.add_to_log_weight_estimates(slp, log_weight_estimate)

            if self.parallelisation == "multi-processing":
                slp_weight_estimate_keys: List[jax.Array] = []
                for slp_ix, slp in tqdm(enumerate(self.active_slps), total=len(self.active_slps), desc=f"Iteration {self.iteration_counter}", position=0):
                    rng_key, slp_inference_key, slp_weight_estimate_key = jax.random.split(rng_key, 3)
                    
                    inference_task = self.make_inference_task(slp, slp_inference_key)
                    
                    # inference_task = InferenceTask(lambda x: x*2, (jax.random.normal(jax.random.PRNGKey(0), (10,)),))

                    exported_fn, in_tree, out_tree = _export_flat(inference_task.f, ("cpu",), (), None)(*inference_task.args)
                    flat_args, _in_tree = tree_flatten(inference_task.args)
                    assert _in_tree == in_tree

                    work_aux = (slp_ix, in_tree, out_tree)
                    work = (exported_fn.serialize(), tuple(flat_args))
                        

                    task_queue.put(((work_aux, work)))
                    slp_weight_estimate_keys.append(slp_weight_estimate_key)

                task_queue.join()

                for _ in range(len(self.active_slps)):
                    (slp_ix, in_tree, out_tree), response = result_queue.get()
                    inference_result = tree_unflatten(out_tree, response)
                    assert isinstance(inference_result, InferenceResult)
                    slp = self.active_slps[slp_ix]
                    self.add_to_inference_results(slp, inference_result)

                # for now we do log_weight_estimate in sequence
                for slp, slp_weight_estimate_key in tqdm(zip(self.active_slps, slp_weight_estimate_keys), total=len(self.active_slps)):
                    log_weight_estimate = self.estimate_log_weight(slp, slp_weight_estimate_key)
                    self.add_to_log_weight_estimates(slp, log_weight_estimate)

                

            rng_key, update_key = jax.random.split(rng_key)
            self.update_active_slps(self.active_slps, self.inactive_slps, self.inference_results, self.log_weight_estimates, update_key)
        
        task_queue.shutdown()
        result_queue.shutdown()
        for t in threads:
            t.join()

        combined_result = self.combine_results(self.inference_results, self.log_weight_estimates)
        
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