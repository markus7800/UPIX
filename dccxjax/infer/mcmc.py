import jax.experimental
from ..types import Trace, PRNGKey, FloatArray, IntArray
import jax
import jax.numpy as jnp
from typing import Callable, Generator, Any, Tuple, Optional, List, NamedTuple, TypeVar
from .gibbs_model import GibbsModel
from .variable_selector import VariableSelector
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dccxjax.core import SLP
from ..utils import JitVariationTracker, maybe_jit_warning, to_shaped_arrays, broadcast_jaxtree
from time import time

from tqdm.auto import tqdm as tqdm_auto

__all__ = [
    "InferenceStep",
    "Gibbs",
    "mcmc",
    "vectorise_kernel_over_chains",
    "init_inference_infos",
    "init_inference_infos_for_chains"
]

InferenceInfo = NamedTuple
InferenceInfos = List[InferenceInfo]

class KernelState(NamedTuple):
    position: Trace
    log_prob: FloatArray
    info: Optional[InferenceInfo]
    
Kernel = Callable[[PRNGKey,FloatArray,KernelState],KernelState]

class MCMCState(NamedTuple):
    iteration: IntArray # scalar
    temperature: FloatArray # scalar 0 ... prior, 1 ... joint
    position: Trace
    log_prob: FloatArray
    infos: InferenceInfos


class InferenceAlgorithm(ABC):
    @abstractmethod
    def make_kernel(self, gibbs_model: GibbsModel, step_number: int, collect_inferenence_info: bool) -> Kernel:
        raise NotImplementedError
    
    @abstractmethod
    def init_info(self) -> InferenceInfo:
        raise NotImplementedError
    

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


MCMC_COLLECT_TYPE = TypeVar("MCMC_COLLECT_TYPE")
MCMCKernel = Callable[[MCMCState,PRNGKey],Tuple[MCMCState,MCMC_COLLECT_TYPE]]

def get_inference_regime_mcmc_step_for_slp(slp: SLP, regime: InferenceRegime, *,
    collect_inference_info: bool = False,
    vectorised: bool = True,
    return_map: Callable[[MCMCState], MCMC_COLLECT_TYPE] = lambda _: None) -> MCMCKernel[MCMC_COLLECT_TYPE]:
    
    kernels: List[Kernel] = []
    for step_number, inference_step in enumerate(regime):
        gibbs_model = GibbsModel(slp, inference_step.variable_selector)
        kernels.append(inference_step.algo.make_kernel(gibbs_model, step_number, collect_inference_info))
    
    jit_tracker = JitVariationTracker(f"_mcmc_step for {slp.short_repr()}")
    @jax.jit
    def one_step(state: MCMCState, rng_key: PRNGKey) -> Tuple[MCMCState,MCMC_COLLECT_TYPE]:
        maybe_jit_warning(jit_tracker, str(to_shaped_arrays(state)))
        
        # rng_key = state.rng_key
        position = state.position
        log_prob = state.log_prob
        infos = state.infos

        for i, kernel in enumerate(kernels):
            rng_key, kernel_key = jax.random.split(rng_key)
            kernel_state = KernelState(position, log_prob, infos[i] if collect_inference_info else None)

            if vectorised:
                # for some reason this is significantly faster
                # re-compilation time for different number of chains should be okay since sub-kernels are always cached
                t = state.log_prob.shape[0]
                kernel_keys = jax.random.split(kernel_key, t)
                new_kernel_state = jax.vmap(kernel, in_axes=(0,None,0))(kernel_keys, state.temperature, kernel_state) # (1)
            else:
                new_kernel_state = kernel(kernel_key, state.temperature, kernel_state) # (2)

            position = new_kernel_state.position
            log_prob = new_kernel_state.log_prob
            if collect_inference_info:
                assert infos is not None and new_kernel_state.info is not None
                infos[i] = new_kernel_state.info

        return MCMCState(state.iteration + 1, state.temperature, position, log_prob, infos), return_map(state)

    return one_step

# this with (2) seems to be a lot slower than (1)
# kernel had to be created with vectorised=False
def vectorise_kernel_over_chains(kernel: MCMCKernel[MCMC_COLLECT_TYPE]) -> MCMCKernel[MCMC_COLLECT_TYPE]:
    jit_tracker = JitVariationTracker(f"vectorise <Kernel {hex(id(kernel))}>")
    @jax.jit
    def _vectorised_kernel(state: MCMCState, rng_key: PRNGKey) -> Tuple[MCMCState,MCMC_COLLECT_TYPE]:
        n_chains = state.log_prob.shape[0]
        maybe_jit_warning(jit_tracker, f"n_chains={n_chains}")
        chain_keys = jax.random.split(rng_key, n_chains)
        axes = (MCMCState(None,None,0,0,0),0) # type: ignore
        # can use the same for in and out, but once refers to rng_keys and once to mcmc collect object
        return jax.vmap(kernel, in_axes=axes, out_axes=axes)(state, chain_keys)
    return _vectorised_kernel

class ProgressbarManager:
    def __init__(self, num_samples: int) -> None:
        self.tqdm_bar: Optional[tqdm_auto] = None
        self.num_samples = num_samples

    def start_progress(self):
        self.tqdm_bar = tqdm_auto(range(self.num_samples), position=0)
        self.tqdm_bar.set_description("Compiling... ", refresh=True)

    def _init_tqdm(self):
        if self.tqdm_bar is not None:
            self.tqdm_bar.set_description(f"Running MCMC", refresh=True)
            # t1 = time()
            # tqdm_auto.write(f"Compile time {t1-t0:.3f}s")

    def _update_tqdm(self, increment):
        if self.tqdm_bar is not None:
            increment = int(increment)
            self.tqdm_bar.update(increment)

    def _close_tqdm(self, increment):
        if self.tqdm_bar is not None:
            increment = int(increment)
            self.tqdm_bar.update(increment)
            self.tqdm_bar.close()
            self.tqdm_bar = None

# adapted form numpyro/util.py
def add_progress_bar(num_samples: int, kernel: MCMCKernel[MCMC_COLLECT_TYPE]) -> Tuple[ProgressbarManager,MCMCKernel[MCMC_COLLECT_TYPE]]:

    if num_samples > 100:
        print_rate = int(num_samples / 100)
    else:
        print_rate = 1


    remainder = num_samples % print_rate


    progressbar_mngr = ProgressbarManager(num_samples)
    # t0 = 0

    
    def _update_progress_bar(iter_num: jax.Array):
        # nonlocal t0
        # t0 = time()
        
        iter_num = iter_num + 1 # all chains are at the same iteration, init iteration=0
        _ = jax.lax.cond(
            iter_num == 1,
            lambda _: jax.experimental.io_callback(progressbar_mngr._init_tqdm, None),
            lambda _: None,
            operand=None,
        )
        _ = jax.lax.cond(
            iter_num % print_rate == 0,
            lambda _: jax.experimental.io_callback(progressbar_mngr._update_tqdm, None, print_rate),
            lambda _: None,
            operand=None,
        )
        _ = jax.lax.cond(
            iter_num == num_samples,
            lambda _: jax.experimental.io_callback(progressbar_mngr._close_tqdm, None, remainder),
            lambda _: None,
            operand=None,
        )
    
    def wrapped_kernel(state: MCMCState, rng_key: PRNGKey):
        _update_progress_bar(state.iteration) # NOTE: we don't have to return something for this to work?
        return kernel(state, rng_key)
    
    return progressbar_mngr, wrapped_kernel



def init_inference_infos(regime: InferenceRegime) -> InferenceInfos:
    return [step.algo.init_info() for step in regime]
def init_inference_infos_for_chains(regime: InferenceRegime, n_chains: int) -> InferenceInfos:
    return broadcast_jaxtree([step.algo.init_info() for step in regime], (n_chains,))

def get_initial_inference_state(slp: SLP, regime: InferenceRegime, n_chains: int, collect_inference_info: bool):
    inference_info: InferenceInfos = [step.algo.init_info() for step in regime] if collect_inference_info else []

    # add leading dimension by broadcasting, i.e. X of shape (m,n,...) has now shape (n_chains,m,n,...)
    # and X[i,m,n,...] = X[j,m,n,...] for all i, j
    return broadcast_jaxtree(InferenceCarry(0, 1., InferenceState(slp.decision_representative, slp.log_prob(slp.decision_representative)), inference_info), (n_chains,))

    result, lp = coordinate_ascent(slp, 0.1, 1000, n_chains, jax.random.PRNGKey(0))
    assert not jnp.isinf(lp).any()
    print("HERE", result["start"], lp)
    return InferenceState(result)


def mcmc(slp: SLP, regime: InferenceRegime, n_samples: int, n_chains: int, rng_key: PRNGKey, collect_states: bool = True, collect_inference_info: bool = False):
    
    keys = jax.random.split(rng_key, n_samples)

    mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, regime, n_chains, collect_inference_info, return_map=lambda x: x.state.position if collect_states else None)

    # mcmc_step = add_progress_bar(n_samples, mcmc_step)

    init = get_initial_inference_state(slp, regime, n_chains, collect_inference_info)

    last_state, all_positions = jax.lax.scan(mcmc_step, init, keys)
 
    return all_positions if collect_states else last_state.state.position # TODO

