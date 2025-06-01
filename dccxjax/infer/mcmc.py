import jax.experimental
from ..types import Trace, PRNGKey, FloatArray, IntArray, StackedTrace, BoolArray
import jax
import jax.numpy as jnp
from typing import Callable, Generator, Any, Tuple, Optional, List, NamedTuple, TypeVar, Generic, TypedDict, Set, Dict
from .gibbs_model import GibbsModel
from .variable_selector import VariableSelector
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dccxjax.core.model_slp import SLP, AnnealingMask
from ..utils import JitVariationTracker, maybe_jit_warning, pprint_dtype_shape_of_tree, broadcast_jaxtree
from time import time
from multipledispatch import dispatch
from jax.flatten_util import ravel_pytree
from .variable_selector import AllVariables

from tqdm.auto import tqdm as tqdm_auto

__all__ = [
    "MCMCRegime",
    "MCMCSteps",
    "MCMCStep",
    "MCMC",
    "vectorise_kernel_over_chains",
    "init_inference_infos",
    "init_inference_infos_for_chains",
    "summarise_mcmc_info",
    "summarise_mcmc_infos",
    "pprint_mcmc_regime"
]

InferenceInfo = NamedTuple
InferenceInfos = List[InferenceInfo]

@dispatch(Any, int)
def summarise_mcmc_info(info, n_samples: int) -> str:
    raise NotImplementedError

def summarise_mcmc_infos(infos: InferenceInfos, n_samples: int):
    info_strs = []
    for step, info in enumerate(infos):
        info_strs.append(f"Step {step}: {summarise_mcmc_info(info, n_samples)}")
    return "\n".join(info_strs)

class CarryStats(TypedDict, total=False):
    position: Trace
    log_prob: FloatArray
    unconstrained_position: Trace
    unconstrained_log_prob: FloatArray

def map_carry_stats(carry_stats: CarryStats, gibbs_model: GibbsModel, temperature: FloatArray, data_annealing: AnnealingMask, new_stats: Set[str]):
    new_carry_stats = CarryStats()
    if "position" in new_stats and "position" not in carry_stats:
        assert "unconstrained_position" in carry_stats 
        new_carry_stats["position"] = gibbs_model.slp.transform_to_constrained(carry_stats["unconstrained_position"])
    if "log_prob" in new_stats and "log_prob" not in carry_stats:
        assert "position" in new_carry_stats 
        new_carry_stats["log_prob"] = gibbs_model.tempered_log_prob(temperature, data_annealing)(new_carry_stats["position"])

    if "unconstrained_position" in new_stats and "unconstrained_position" not in carry_stats:
        assert "position" in carry_stats 
        new_carry_stats["unconstrained_position"] = gibbs_model.slp.transform_to_unconstrained(carry_stats["position"])
    if "unconstrained_log_prob" in new_stats and "unconstrained_log_prob" not in carry_stats:
        assert "unconstrained_position" in new_carry_stats 
        new_carry_stats["unconstrained_log_prob"] = gibbs_model.tempered_unconstrained_log_prob(temperature, data_annealing)(new_carry_stats["unconstrained_position"])

    for stat in new_stats:
        if stat not in new_carry_stats:
            if stat in carry_stats:
                new_carry_stats[stat] = carry_stats[stat]
            else:
                raise Exception(f"Do not know how to fill stat {stat} with {sorted(carry_stats.keys())}")
        
    return new_carry_stats
            

class KernelState(NamedTuple):
    carry_stats: CarryStats
    info: Optional[InferenceInfo]
    
Kernel = Callable[[PRNGKey,FloatArray,AnnealingMask,KernelState],KernelState]


class MCMCState(NamedTuple):
    iteration: IntArray # scalar
    temperature: FloatArray # scalar 0 ... prior, 1 ... joint
    data_annealing: AnnealingMask
    position: Trace
    log_prob: FloatArray
    carry_stats: CarryStats
    infos: Optional[InferenceInfos]


class MCMCInferenceAlgorithm(ABC):
    unconstrained: bool

    @abstractmethod
    def make_kernel(self, gibbs_model: GibbsModel, step_number: int, collect_inferenence_info: bool) -> Kernel:
        raise NotImplementedError
    
    @abstractmethod
    def init_info(self) -> InferenceInfo:
        raise NotImplementedError
    
    def requires_stats(self) -> Set[str]:
        return {"unconstrained_position", "unconstrained_log_prob"} if self.unconstrained else {"position", "log_prob"}
    
    def provides_stats(self) -> Set[str]:
        return {"unconstrained_position", "unconstrained_log_prob"} if self.unconstrained else {"position", "log_prob"}
    
    def default_preprocess_to_flat(self, gibbs_model: GibbsModel, temperature: FloatArray, data_annealing: AnnealingMask, state: KernelState):
        if self.unconstrained:
            assert "unconstrained_position" in state.carry_stats
            assert "unconstrained_log_prob" in state.carry_stats
            current_postion: Trace = state.carry_stats["unconstrained_position"]
            log_prob: FloatArray = state.carry_stats["unconstrained_log_prob"]

        else:
            assert "position" in state.carry_stats
            assert "log_prob" in state.carry_stats
            current_postion: Trace = state.carry_stats["position"]
            log_prob: FloatArray = state.carry_stats["log_prob"]

        X, Y = gibbs_model.split_trace(current_postion)
        gibbs_model.set_Y(Y)
        X_flat, unravel_fn = ravel_pytree(X)

        if self.unconstrained:
            _tempered_log_prob_fn = gibbs_model.unraveled_unconstrained_tempered_log_prob(temperature, data_annealing, unravel_fn)
        else:
            _tempered_log_prob_fn = gibbs_model.unraveled_tempered_log_prob(temperature, data_annealing, unravel_fn)

        return X_flat, log_prob, unravel_fn, _tempered_log_prob_fn
    
    def default_postprocess_from_flat(self, gibbs_model: GibbsModel, X_flat: jax.Array, log_prob: FloatArray, unravel_fn: Callable[[jax.Array], Trace]):
        next_position = gibbs_model.combine_to_trace(unravel_fn(X_flat), gibbs_model.Y)
        if self.unconstrained:
            return CarryStats(unconstrained_position=next_position, unconstrained_log_prob=log_prob)
        else:
            return CarryStats(position=next_position, log_prob=log_prob)
        


class MCMCRegime(ABC):
    @abstractmethod
    def __iter__(self) -> Generator["MCMCStep", Any, None]:
        pass

class MCMCStep(MCMCRegime):
    def __init__(self, variable_selector: VariableSelector, algo: MCMCInferenceAlgorithm) -> None:
        self.variable_selector = variable_selector
        self.algo = algo
    def __iter__(self):
        yield self
    def description(self, slp: Optional[SLP]) -> str:
        if slp is not None:
            target = sorted([address for address in slp.decision_representative.keys() if self.variable_selector.contains(address)])
        else:
            target = str(self.variable_selector)
        return f"{self.algo} targeting {target}"

class MCMCSteps(MCMCRegime):
    def __init__(self, *subregimes: MCMCRegime) -> None:
        self.subregimes = subregimes
    def __iter__(self):
        for subregime in self.subregimes:
            if isinstance(subregime, MCMCStep):
                yield subregime
            else:
                assert isinstance(subregime, MCMCSteps)
                yield from subregime

class ProgressbarManager:
    def __init__(self, desc: str) -> None:
        self.desc = desc
        self.tqdm_bar: Optional[tqdm_auto] = None
        self.num_samples = 0

    def set_num_samples(self, num_samples: int):
        self.num_samples = num_samples # affects for tqdm bar lenght, not print rate

    def start_progress(self):
        assert self.tqdm_bar is None
        self.tqdm_bar = tqdm_auto(range(self.num_samples), position=0)
        self.tqdm_bar.set_description(f"Compiling {self.desc}... ", refresh=True)

    def _init_tqdm(self, increment):
        if self.tqdm_bar is None: 
            self.tqdm_bar = tqdm_auto(range(self.num_samples), position=0)
        increment = int(increment)
        self.tqdm_bar.set_description(f"  Running {self.desc}", refresh=True)
        self.tqdm_bar.update(increment)

    def _update_tqdm(self, iternum, increment, remainder):
        if self.tqdm_bar is not None:
            iternum = int(iternum)
            increment = int(increment)
            remainder = int(remainder)
            # tqdm_auto.write(f"update tqdm {iternum} inc={increment}")
            if iternum == self.num_samples:
                if remainder == 0:
                    # update and close event happen at same time
                    self.tqdm_bar.update(increment)
                else:
                    self.tqdm_bar.update(remainder)
                self.tqdm_bar.close()
                self.tqdm_bar = None
            else:
                self.tqdm_bar.update(increment)



MCMC_COLLECT_TYPE = TypeVar("MCMC_COLLECT_TYPE")
MCMCKernel = Callable[[MCMCState,PRNGKey],Tuple[MCMCState,MCMC_COLLECT_TYPE]]

def _get_sym_for_stat(stat: str, curr_stats: Set[str], next_stats: Set[str]) -> str:
    if stat in curr_stats and stat in next_stats:
        return "="
    elif stat in curr_stats:
        return "-"
    else:
        return "+"

def pprint_mcmc_regime(regime: MCMCRegime, slp: Optional[SLP]=None):
    regime_steps: List[MCMCStep] = list(regime)

    init_carry_stat_names = (regime_steps[0].algo.requires_stats() & regime_steps[-1].algo.provides_stats()) - {"position", "log_prob"}

    print("MCMC Regime:")
    curr_stats = init_carry_stat_names | {"position", "log_prob"}
    s = "Call stats: "
    for stat in sorted(curr_stats):
        s += stat + " "
    print("\t", s)

    next_stats = regime_steps[0].algo.requires_stats()
    s = "Init: "
    for stat in sorted(curr_stats | next_stats):
        s += _get_sym_for_stat(stat, curr_stats, next_stats) + stat + " "
    print("\t", s)

    for i in range(0, len(regime_steps)-1):
        curr_step = regime_steps[i]
        next_step = regime_steps[i+1]
        curr_stats = curr_step.algo.provides_stats()
        next_stats = next_step.algo.requires_stats()
        s = "Transfer: "
        for stat in sorted(curr_stats | next_stats):
            s += _get_sym_for_stat(stat, curr_stats, next_stats) + stat + " "
        print("\t", f"Step {i}.", curr_step.description(slp))
        print("\t", s)
    print("\t", f"Step {len(regime_steps)-1}.", regime_steps[-1].description(slp))

    curr_stats = regime_steps[-1].algo.provides_stats()
    next_stats = regime_steps[0].algo.requires_stats() | {"position", "log_prob"}
    s = "Wrap: "
    for stat in sorted(curr_stats | next_stats):
        s += _get_sym_for_stat(stat, curr_stats, next_stats) + stat + " "
    print("\t", s)

def get_mcmc_kernel(
        slp: SLP, regime: MCMCRegime, *,
        collect_inference_info: bool = False, 
        vectorised: bool = True,
        return_map: Callable[[MCMCState], MCMC_COLLECT_TYPE] = lambda _: None) -> Tuple[MCMCKernel[MCMC_COLLECT_TYPE], Set[str]]:
    
    regime_steps: List[MCMCStep] = list(regime)
    kernels: List[Kernel] = []
    for step_number, inference_step in enumerate(regime_steps):
        gibbs_model = GibbsModel(slp, inference_step.variable_selector)
        kernels.append(inference_step.algo.make_kernel(gibbs_model, step_number, collect_inference_info))
    
    full_model = GibbsModel(slp, AllVariables())

    # they have to be initialised before kernel is run
    mcmc_carry_stat_names = (regime_steps[0].algo.requires_stats() & regime_steps[-1].algo.provides_stats()) - {"position", "log_prob"}

    jit_tracker = JitVariationTracker(f"_mcmc_step for {slp.short_repr()}")
    @jax.jit
    def _one_step(state: MCMCState, rng_key: PRNGKey) -> Tuple[MCMCState,MCMC_COLLECT_TYPE]:
        maybe_jit_warning(jit_tracker, str(pprint_dtype_shape_of_tree(state)))
        
        # rng_key = state.rng_key
        position = state.position
        log_prob = state.log_prob

        carry_stats: CarryStats = state.carry_stats
        assert carry_stats.keys() == mcmc_carry_stat_names, f"{carry_stats.keys()} does not match {mcmc_carry_stat_names}"
        carry_stats["position"] = position
        carry_stats["log_prob"] = log_prob

        infos = state.infos

        _map_carry_stats = jax.vmap(map_carry_stats, in_axes=(0,None,None,None,None)) if vectorised else map_carry_stats

        for i, (step, kernel) in enumerate(zip(regime_steps, kernels)):
            rng_key, kernel_key = jax.random.split(rng_key)

            carry_stats = _map_carry_stats(carry_stats, full_model, state.temperature, state.data_annealing, step.algo.requires_stats())
            assert step.algo.requires_stats() <= carry_stats.keys()
            kernel_state = KernelState(carry_stats, infos[i] if collect_inference_info and infos is not None else None)

            if vectorised:
                # for some reason this is significantly faster
                # re-compilation time for different number of chains should be okay since sub-kernels are always cached
                t = state.log_prob.shape[0]
                if t == 1:
                    kernel_keys = jax.lax.broadcast(kernel_key, (1,))
                else:
                    kernel_keys = jax.random.split(kernel_key, t)
                new_kernel_state = jax.vmap(kernel, in_axes=(0,None,None,0))(kernel_keys, state.temperature, state.data_annealing, kernel_state) # (1)
            else:
                new_kernel_state = kernel(kernel_key, state.temperature, state.data_annealing, kernel_state) # (2)

            carry_stats = new_kernel_state.carry_stats
            assert carry_stats.keys() <= step.algo.provides_stats()

            if collect_inference_info:
                assert infos is not None and new_kernel_state.info is not None
                infos[i] = new_kernel_state.info

        carry_stats = _map_carry_stats(carry_stats, full_model, state.temperature, state.data_annealing, {"position", "log_prob"} | carry_stats.keys())
        assert "position" in carry_stats and "log_prob" in carry_stats
        position = carry_stats["position"]
        log_prob = carry_stats["log_prob"]
                               
        next_carry_stats = CarryStats()
        for stat, carry in carry_stats.items():
            if stat in mcmc_carry_stat_names:
                next_carry_stats[stat] = carry


        return MCMCState(state.iteration + 1, state.temperature, state.data_annealing, position, log_prob, next_carry_stats, infos), return_map(state)

    return _one_step, mcmc_carry_stat_names


# this with (2) seems to be a lot slower than (1)
# kernel had to be created with vectorised=False
def vectorise_kernel_over_chains(kernel: MCMCKernel[MCMC_COLLECT_TYPE]) -> MCMCKernel[MCMC_COLLECT_TYPE]:
    jit_tracker = JitVariationTracker(f"vectorise <Kernel {hex(id(kernel))}>")
    @jax.jit
    def _vectorised_kernel(state: MCMCState, rng_key: PRNGKey) -> Tuple[MCMCState,MCMC_COLLECT_TYPE]:
        n_chains = state.log_prob.shape[0]
        maybe_jit_warning(jit_tracker, f"n_chains={n_chains}")
        chain_keys = jax.random.split(rng_key, n_chains)
        axes = (MCMCState(None,None,None,0,0,0,0),0) # type: ignore
        # can use the same for in and out, but once refers to rng_keys and once to mcmc collect object
        return jax.vmap(kernel, in_axes=axes, out_axes=axes)(state, chain_keys)
    return _vectorised_kernel

# adapted form numpyro/util.py
def _add_progress_bar(kernel: Callable, progressbar_mngr: ProgressbarManager, num_samples: int) -> Callable:
    print_rate = max(int(num_samples / 100), 1)

    remainder = num_samples % print_rate

    def _update_progress_bar(iter_num: jax.Array):
        # nonlocal t0
        # t0 = time()
        
        iter_num = iter_num + 1 # all chains are at the same iteration, init iteration=0
        _ = jax.lax.cond(
            (iter_num % print_rate == 0) | (iter_num == num_samples),
            lambda _: jax.experimental.io_callback(progressbar_mngr._update_tqdm, None, iter_num, print_rate, remainder),
            lambda _: None,
            operand=None,
        )
    
    def wrapped_kernel(state, data):
        result = kernel(state, data)
        _update_progress_bar(state.iteration) # NOTE: we don't have to return something for this to work?
        return result
    
    return wrapped_kernel

def get_mcmc_scan_with_progressbar(kernel: MCMCKernel[MCMC_COLLECT_TYPE], progressbar_mngr: ProgressbarManager, num_samples: int) -> Callable[[MCMCState, PRNGKey], Tuple[MCMCState,MCMC_COLLECT_TYPE]]:
    progressbar_mngr.set_num_samples(num_samples)
    kernel_with_bar = _add_progress_bar(kernel, progressbar_mngr, num_samples)
    def scan_with_bar(init: MCMCState, xs: PRNGKey) -> Tuple[MCMCState, MCMC_COLLECT_TYPE]:
        progressbar_mngr.start_progress()
        jax.experimental.io_callback(progressbar_mngr._init_tqdm, None, init.iteration)
        return jax.lax.scan(kernel_with_bar, init, xs)
    return jax.jit(scan_with_bar)

def get_mcmc_scan_without_progressbar(kernel: MCMCKernel[MCMC_COLLECT_TYPE]) -> Callable[[MCMCState, PRNGKey], Tuple[MCMCState,MCMC_COLLECT_TYPE]]:
    def scan_without_bar(init: MCMCState, xs: PRNGKey) -> Tuple[MCMCState, MCMC_COLLECT_TYPE]:
        return jax.lax.scan(kernel, init, xs)
    return jax.jit(scan_without_bar)


class MCMC(Generic[MCMC_COLLECT_TYPE]):
    def __init__(self,
        slp: SLP,
        regime: MCMCRegime,
        n_chains: int,
        *,
        collect_inference_info: bool = False,
        vectorised: bool = True,
        return_map: Callable[[MCMCState], MCMC_COLLECT_TYPE] = lambda _: None,
        temperature: FloatArray = jnp.array(1.0),
        data_annealing: AnnealingMask = dict(),
        reuse_kernel: Optional[MCMCKernel[MCMC_COLLECT_TYPE]] = None,
        reuse_kernel_init_carry_stat_names: Set[str] = set(),
        progress_bar: bool = False) -> None:
        
        self.slp = slp
        self.regime = regime
        self.n_chains = n_chains
        self.progress_bar = progress_bar
        self.progress_bar_mngr = ProgressbarManager("MCMC for "+self.slp.formatted())

        self.temperature = temperature
        self.data_annealing = data_annealing

        self.collect_inference_info = collect_inference_info
        self.vectorised = vectorised

        if reuse_kernel is not None:
            kernel = reuse_kernel
            init_carry_stat_names = reuse_kernel_init_carry_stat_names
        else:
            kernel, init_carry_stat_names = get_mcmc_kernel(slp, regime, collect_inference_info=collect_inference_info, vectorised=vectorised, return_map=return_map)

        self.kernel: MCMCKernel[MCMC_COLLECT_TYPE] = kernel
        self.init_carry_stat_names: Set[str] = init_carry_stat_names

        self.cached_mcmc_scan: Optional[Callable[[MCMCState,PRNGKey],Tuple[MCMCState,MCMC_COLLECT_TYPE]]] = None

    def run(self, rng_key: PRNGKey, init_positions: StackedTrace, log_prob: Optional[FloatArray] = None, *, n_samples_per_chain: int) -> Tuple[MCMCState, MCMC_COLLECT_TYPE]:
        assert init_positions.T == self.n_chains
        if log_prob is None:
            log_prob = jax.vmap(self.slp.log_prob, in_axes=(0,None,None))(init_positions.data, self.temperature, self.data_annealing)
        else:
            assert log_prob.shape == (self.n_chains,)

        infos = init_inference_infos_for_chains(self.regime, self.n_chains) if self.collect_inference_info else None

        carry_stats = CarryStats(position=init_positions.data, log_prob=log_prob)

        carry_stats = jax.vmap(map_carry_stats, in_axes=(0,None,None,None,None))(carry_stats, GibbsModel(self.slp, AllVariables()), self.temperature, self.data_annealing, self.init_carry_stat_names)

        initial_state = MCMCState(jnp.array(0, int), self.temperature, self.data_annealing, init_positions.data, log_prob, carry_stats, infos)

        return self.continue_run(rng_key, initial_state, n_samples_per_chain=n_samples_per_chain)
        
    def continue_run(self, rng_key: PRNGKey, state: MCMCState, *, n_samples_per_chain: int, reset_info: bool=True):
        keys = jax.random.split(rng_key, n_samples_per_chain)
        if reset_info:
            infos = init_inference_infos_for_chains(self.regime, self.n_chains) if self.collect_inference_info else None
            state = MCMCState(jnp.array(0, int), self.temperature, self.data_annealing, state.position, state.log_prob, state.carry_stats, infos)

        if self.cached_mcmc_scan:
            self.progress_bar_mngr.set_num_samples(n_samples_per_chain)
            last_state, return_values = self.cached_mcmc_scan(state, keys)
        else:
            scan_fn = (
                get_mcmc_scan_with_progressbar(self.kernel, self.progress_bar_mngr, n_samples_per_chain) # print_rate will be based on n_samples_per_chain
                if self.progress_bar else
                get_mcmc_scan_without_progressbar(self.kernel)
            )
            self.cached_mcmc_scan = scan_fn
            last_state, return_values = scan_fn(state, keys)

        return last_state, return_values

def init_inference_infos(regime: MCMCRegime) -> InferenceInfos:
    return [step.algo.init_info() for step in regime]

def init_inference_infos_for_chains(regime: MCMCRegime, n_chains: int) -> InferenceInfos:
    return broadcast_jaxtree([step.algo.init_info() for step in regime], (n_chains,))


