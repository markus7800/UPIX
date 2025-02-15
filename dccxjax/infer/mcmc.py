from ..types import Trace, PRNGKey
import jax
from typing import Callable, Generator, Any, Tuple, Optional, List
from .gibbs_model import GibbsModel
from .variable_selector import VariableSelector
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dccxjax.core import SLP
from ..utils import maybe_jit_warning, to_shaped_arrays


__all__ = [
    "InferenceStep",
    "Gibbs",
    "unstack_chains",
    "n_samples_for_stacked_chains",
    "n_samples_for_unstacked_chains",
]

@jax.tree_util.register_dataclass
@dataclass
class InferenceState(object):
    position: Trace
    
    
Kernel = Callable[[PRNGKey,InferenceState],InferenceState]

class InferenceAlgorithm(ABC):
    @abstractmethod
    def make_kernel(self, gibbs_model: GibbsModel, step_number: int) -> Kernel:
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


# TODO: track compile time + progressbar
def get_inference_regime_mcmc_step_for_slp(slp: SLP, regime: InferenceRegime, n_chains: int, collect_states: bool) -> Callable[[InferenceState,PRNGKey],Tuple[InferenceState,Optional[Trace]]]:
    kernels: List[Kernel] = []
    for step_number, inference_step in enumerate(regime):
        gibbs_model = GibbsModel(slp, inference_step.variable_selector)
        kernels.append(inference_step.algo.make_kernel(gibbs_model, step_number))

    @jax.jit
    def one_step(state: InferenceState, rng_key: PRNGKey) -> Tuple[InferenceState,Optional[Trace]]:
        maybe_jit_warning(None, "", "_mcmc_step", slp.short_repr(), to_shaped_arrays(state))
        for kernel in kernels:
            rng_key, kernel_key = jax.random.split(rng_key)
            kernel_keys = jax.random.split(kernel_key, n_chains)
            state = jax.vmap(kernel)(kernel_keys, state)
        state = InferenceState(state.position)
        return state, state.position if collect_states else None
    
    return one_step




def get_initial_inference_state(slp: SLP, n_chains: int):
    # add leading dimension by broadcasting, i.e. X of shape (m,n,...) has now shape (n_chains,m,n,...)
    # and X[i,m,n,...] = X[j,m,n,...] for all i, j
    return InferenceState(jax.tree_map(lambda x: jax.lax.broadcast(x, (n_chains,)), slp.decision_representative))

    result, lp = coordinate_ascent(slp, 0.1, 1000, n_chains, jax.random.PRNGKey(0))
    assert not jnp.isinf(lp).any()
    print("HERE", result["start"], lp)
    return InferenceState(result)


def mcmc(slp: SLP, regime: InferenceRegime, n_samples: int, n_chains: int, rng_key: PRNGKey, collect_states: bool = True):
    
    keys = jax.random.split(rng_key, n_samples)

    mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, regime, n_chains, collect_states)

    init = get_initial_inference_state(slp, n_chains)

    last_state, all_positions = jax.lax.scan(mcmc_step, init, keys)
 
    return all_positions if collect_states else last_state.position


def _unstack_chains(values: jax.Array):
    shape = values.shape
    assert len(shape) >= 2
    var_dim = () if len(shape) < 3 else (shape[2],)
    n_samples = shape[0]
    n_chains = shape[1]
    return jax.lax.reshape(values, (n_samples * n_chains, *var_dim))

def unstack_chains(Xs: Trace) -> Trace:
    return jax.tree_map(_unstack_chains, Xs)

def n_samples_for_stacked_chains(Xs: Trace):
    _, some_entry = next(iter(Xs.items()))
    return some_entry.shape[0] * some_entry.shape[1]

def n_samples_for_unstacked_chains(Xs: Trace):
    _, some_entry = next(iter(Xs.items()))
    return some_entry.shape[0]
