import jax
import jax.numpy as jnp
from typing import Dict, Optional, List, Callable, Any, NamedTuple, Generic, TypeVar, Tuple, cast
from upix.core import SLP, Model, sample_from_prior, slp_from_decision_representative
from upix.types import Trace, PRNGKey, FloatArray, IntArray
from upix.utils import get_backend, get_default_device, log_warn
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from upix.infer.dcc.dcc_types import InferenceResult, LogWeightEstimate, JaxTask, InferenceTask, EstimateLogWeightTask
import os
import threading
from queue import Queue
from upix.infer.dcc.cpu_multiprocess import start_worker_process, start_worker_thread
from upix.parallelisation import ParallelisationConfig, ParallelisationType, VectorisationType, is_sequential, is_parallel
import time

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
    
    def sprint(self, *, sortkey: str = "logweight"):
        assert sortkey in ("log_weight", "slp")
        log_Z_normaliser = self.get_log_weight_normaliser()
        s = "BaseDCCResult {\n"
        slp_log_weights_list = self.get_log_weights_sorted(sortkey)
        for slp, log_weight in slp_log_weights_list:
            s += f"\t{slp.formatted()}: with prob={jnp.exp(log_weight - log_Z_normaliser).item():.6f}, log_Z={log_weight.item():6f}\n"
        s += "}\n"
        return s
        
    def pprint(self, *, sortkey: str = "logweight"):
        print(self.sprint(sortkey=sortkey))
    
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
    
def append_info(info: str, new_info: str):
    if info == "":
        return new_info
    else:
        return info + "\n" + new_info
    
DCC_RESULT_TYPE = TypeVar("DCC_RESULT_TYPE")
class AbstractDCC(ABC, Generic[DCC_RESULT_TYPE]):
    def __init__(self, model: Model, *ignore, verbose=0, **config_kwargs) -> None:
        if ignore:
            raise TypeError
        
        self.model = model
        self.config: Dict[str, Any] = config_kwargs

        self.verbose = verbose
        
        self.inference_time = 0.
        self.logweight_estimate_time = 0.
        self.total_time = 0.
        
        self.inference_method_cache: Dict[SLP, Any] = dict()
        
        self.share_progress_bar: bool = self.config.get("share_progress_bar", True)
        self.shared_progress_bar: Optional[tqdm] = None
        self.disable_progress: bool = self.config.get("disable_progress", False)

        self.debug_memory: bool = self.config.get("debug_memory", False)

        self.pconfig: ParallelisationConfig = self.config.get("parallelisation", ParallelisationConfig())

    @abstractmethod
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: PRNGKey):
        raise NotImplementedError
    
    @abstractmethod
    def make_inference_task(self, slp: SLP, rng_key: PRNGKey) -> InferenceTask:
        raise NotImplementedError

    @abstractmethod
    def make_estimate_log_weight_task(self, slp: SLP, rng_key: PRNGKey) -> EstimateLogWeightTask:
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
        
    def maybe_write_info(self, info: str | None):
        if self.verbose > 1 and info is not None and info != "":
            tqdm.write(info)
    
    def run(self, rng_key: PRNGKey):

        self.active_slps: List[SLP] = []
        self.inactive_slps: List[SLP] = []

        task_queue = Queue()
        result_queue = Queue()
        threads: List[threading.Thread] = []

        tqdm.write(f"Start DCC:")
        dcc_t0 = time.monotonic()
        
        _dstr = lambda i, d: str(d) + (" (unused)" if i >= self.pconfig.num_workers else "")
        devices_str = "\n    " + "\n    ".join(map(lambda t: _dstr(*t), enumerate(jax.devices())))
        batch_str = ', vmap_batched='+str(self.pconfig.vmap_batch_size) if self.pconfig.vmap_batch_size > 0 else ""
        if is_sequential(self.pconfig) and self.pconfig.vectorisation == VectorisationType.LocalVMAP:
            tqdm.write(f"parallelisation=Sequential(local vmap{batch_str}, device={get_default_device()})")
        if is_sequential(self.pconfig) and self.pconfig.vectorisation == VectorisationType.GlobalVMAP:
            tqdm.write(f"parallelisation=Sequential(global vmap{batch_str}, device={get_default_device()})")
        if is_sequential(self.pconfig) and self.pconfig.vectorisation == VectorisationType.PMAP:
            tqdm.write(f"parallelisation=Sequential(pmap{batch_str}, \n  devices={devices_str}\n  #workers={self.pconfig.num_workers}\n)")
        if is_sequential(self.pconfig) and self.pconfig.vectorisation == VectorisationType.LocalSMAP:
            tqdm.write(f"parallelisation=Sequential(local smap{batch_str},\n  devices={devices_str}\n #workers={self.pconfig.num_workers}\n)")
        if is_sequential(self.pconfig) and self.pconfig.vectorisation == VectorisationType.GlobalSMAP:
            tqdm.write(f"parallelisation=Sequential(global smap{batch_str},\n  devices={devices_str}\n  #workers={self.pconfig.num_workers}\n)")
        if self.pconfig.parallelisation == ParallelisationType.MultiProcessingCPU:
            tqdm.write(f"parallelisation=MultiProcessingCPU(#workers={self.pconfig.num_workers}{batch_str})")
            backend_platform = get_backend().devices()[0].platform
            if backend_platform != "cpu":
                raise Exception(f"Using MultiProcessingCPU parallelisation, but backend platform ({backend_platform}) is not CPU!")
            for i in range(self.pconfig.num_workers):
                t = threading.Thread(target=start_worker_process, args=(task_queue, result_queue, i, self.pconfig), daemon=True)
                t.start()
                threads.append(t)
        if self.pconfig.parallelisation == ParallelisationType.MultiThreadingJAXDevices:
            batch_str = '\nvmap_batched='+str(self.pconfig.vmap_batch_size) if self.pconfig.vmap_batch_size > 0 else ""
            tqdm.write(f"parallelisation=MultiThreadingJAXDevices(\ndevices={devices_str}{batch_str}\n)")
            assert self.pconfig.num_workers <= len(jax.devices())
            for i in range(self.pconfig.num_workers):
                t = threading.Thread(target=start_worker_thread, args=(task_queue, result_queue, i, self.pconfig), daemon=True)
                t.start()
                threads.append(t)

        self.inference_method_cache: Dict[SLP, Any] = dict()

        self.inference_results: Dict[SLP, List[InferenceResult]] = dict()
        self.log_weight_estimates: Dict[SLP, List[LogWeightEstimate]] = dict()        

        init_slps_t0 = time.monotonic()
        rng_key, init_key = jax.random.split(rng_key)
        self.initialise_active_slps(self.active_slps, self.inactive_slps, init_key)
        init_slps_t1 = time.monotonic()
        tqdm.write(f"Initialised SLPs in {init_slps_t1-init_slps_t0:.3f}s.")

        self.iteration_counter = 0
        self.inference_time = 0.
        self.logweight_estimate_time = 0.
        
        if is_sequential(self.pconfig) and self.share_progress_bar:
            # set bar_format to make bar completely invisible at beginning
            self.shared_progress_bar = tqdm(total=0, position=0, leave=False, bar_format="{bar}", disable=self.disable_progress)
            self.shared_progress_bar.bar_format = "{l_bar}{bar}{r_bar}"
            outer_bar = tqdm(total=0, position=1, leave=False, desc="Iteration 0", mininterval=0, disable=self.disable_progress)
        else:
            outer_bar = tqdm(total=0, position=0, leave=False, desc="Iteration 0", mininterval=0, disable=self.disable_progress)

        while len(self.active_slps) > 0:
            t0 = time.monotonic()
            
            self.iteration_counter += 1
            if self.debug_memory:
                jax.profiler.save_device_memory_profile(f"memory_{self.iteration_counter}.prof")

            if is_sequential(self.pconfig):
                outer_bar.reset(total=len(self.active_slps))
                outer_bar.set_description(f"Iteration {self.iteration_counter}")
                for slp in self.active_slps:
                    rng_key, slp_inference_key, slp_weight_estimate_key = jax.random.split(rng_key, 3)
                    
                    inference_task = self.make_inference_task(slp, slp_inference_key)
                    self.maybe_write_info(inference_task.pre_info())
                    # we do not have to put input to task on device, by default all jitted functions are on device
                    # if we call jitted function with cpu allocated array it will be put on device
                    inference_task_t0 = time.monotonic()
                    inference_result = jax.device_get(inference_task.run()) # device_get puts pytree on host (cpu)
                    jax.block_until_ready(inference_result)
                    elapsed_time_inference = time.monotonic() - inference_task_t0
                    self.inference_time += elapsed_time_inference
                    self.maybe_write_info(append_info(inference_task.post_info(inference_result), f"Finished inference task for {slp.formatted()} in {elapsed_time_inference:.3f}s"))
                    del inference_task # tasks may close over device allocated arrays
                    self.add_to_inference_results(slp, inference_result)
                    outer_bar.update(0.5)

                    log_weight_estimate_task = self.make_estimate_log_weight_task(slp, slp_weight_estimate_key)
                    self.maybe_write_info(log_weight_estimate_task.pre_info())
                    logweigth_task_t0 = time.monotonic()
                    log_weight_estimate = jax.device_get(log_weight_estimate_task.run())
                    jax.block_until_ready(log_weight_estimate)
                    elapsed_time_logweith_estimate = time.monotonic() - logweigth_task_t0
                    self.logweight_estimate_time += elapsed_time_logweith_estimate
                    self.maybe_write_info(append_info((log_weight_estimate_task.post_info(log_weight_estimate)), f"Finished logweight estimation task for {slp.formatted()} in {elapsed_time_logweith_estimate:.3f}s"))
                    del log_weight_estimate_task
                    self.add_to_log_weight_estimates(slp, log_weight_estimate)
                    outer_bar.update(0.5)

            if is_parallel(self.pconfig):
                
                slp_weight_inference_keys: List[jax.Array] = []
                slp_weight_estimate_keys: List[jax.Array] = []
                for _ in self.active_slps:
                    rng_key, slp_inference_key, slp_weight_estimate_key = jax.random.split(rng_key, 3)
                    slp_weight_inference_keys.append(slp_inference_key)
                    slp_weight_estimate_keys.append(slp_weight_estimate_key)
                    
                def make_inference_tasks():
                    for slp_ix, slp in enumerate(self.active_slps):            
                        slp_inference_key = slp_weight_inference_keys[slp_ix]
                        inference_task = self.make_inference_task(slp, slp_inference_key)
                        # make inference task (does not block, but may take a long time)
                        if self.pconfig.parallelisation == ParallelisationType.MultiProcessingCPU:
                            task_queue.put((inference_task.export(), slp_ix))
                        else:        
                            task_queue.put((inference_task, slp_ix))
                        del inference_task
                inference_task_generator = threading.Thread(target=make_inference_tasks, daemon=True)
                inference_task_generator.start()
                if self.pconfig.force_task_order:
                    if self.pconfig.parallelisation == ParallelisationType.MultiProcessingCPU:
                        log_warn(
                            "MultiProcessingCPU parallelisation, but all inference tasks are generated and exported before starting work.\n"
                            "Set force_task_order to False for exporting inference tasks in background."
                        )
                    inference_task_generator.join()

                outer_bar.reset(total=len(self.active_slps))
                outer_bar.set_description(f"Iteration {self.iteration_counter}")
                for _ in range(2*len(self.active_slps)):

                    result, slp_ix, elapsed_time = result_queue.get()
                    slp = self.active_slps[slp_ix]
                    
                    if isinstance(result, InferenceResult):
                        self.add_to_inference_results(slp, jax.device_get(result))
                        self.inference_time += elapsed_time
                        # make logweight estimate task
                        slp_weight_estimate_key = slp_weight_estimate_keys[slp_ix]
                        log_weight_estimate_task = self.make_estimate_log_weight_task(slp, slp_weight_estimate_key)
                        if self.pconfig.parallelisation == ParallelisationType.MultiProcessingCPU:
                            task_queue.put((log_weight_estimate_task.export(), slp_ix))
                        else:
                            task_queue.put((log_weight_estimate_task, slp_ix))
                        del log_weight_estimate_task

                    if isinstance(result, LogWeightEstimate):
                        self.logweight_estimate_time += elapsed_time
                        self.add_to_log_weight_estimates(slp, jax.device_get(result))

                    outer_bar.update(0.5)

            
            
            rng_key, update_key = jax.random.split(rng_key)
            self.update_active_slps(self.active_slps, self.inactive_slps, self.inference_results, self.log_weight_estimates, update_key)
        
        
            t1 = time.monotonic()
            if self.verbose > 0:
                tqdm.write(f"Finished iteration {self.iteration_counter} in {t1-t0:.3f}s.")


        elapsed_time_dcc = time.monotonic() - dcc_t0
        tqdm.write(f"Finished DCC in {elapsed_time_dcc:.3f}s.")
        if self.verbose >= 2:
            tqdm.write(f"Inference time: {self.inference_time:.3f}s, logweight estimate time: {self.logweight_estimate_time:.3f}s")
        outer_bar.close()
        if self.shared_progress_bar is not None:
            self.shared_progress_bar.close()
            
        combined_result = self.combine_results(self.inference_results, self.log_weight_estimates)

        self.total_time = elapsed_time_dcc
        
        return combined_result
    
    def get_timings(self):
        return {
            "inference_time": self.inference_time,
            "logweight_estimate_time": self.logweight_estimate_time,
            "total_time": self.total_time
        }

def initialise_active_slps_from_prior(model: Model, verbose: int, init_n_samples: int, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: PRNGKey, disable_progress: bool):
    if verbose >= 2:
        tqdm.write("Initialise active SLPS.")
    discovered_slps: List[SLP] = []

    # default samples from prior
    for _ in tqdm(range(init_n_samples), desc="Search SLPs from prior", disable=disable_progress):
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