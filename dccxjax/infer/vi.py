import jax.flatten_util
from ..core.samplecontext import GuideContext
from dccxjax.distributions import Transform, Distribution, DIST_SUPPORT, DIST_SUPPORT_LIKE, TransformedDistribution, MultivariateNormal, Normal
from typing import Optional, Sequence, List, Dict, Tuple, NamedTuple, Callable, Generic, TypeVar, Any, cast
from ..types import FloatArray, FloatArrayLike, PRNGKey, Trace
from abc import abstractmethod, ABC
from enum import Enum
import jax
import jax.numpy as jnp
from dccxjax.distributions.constraints import Constraint, real
import numpyro.distributions.transforms as transforms
from ..core.model_slp import Model, SLP
from .mcmc import ProgressbarManager, _add_progress_bar
from .optimizers import OPTIMIZER_STATE, Optimizer
from math import prod
from .gibbs_model import GibbsModel
import jax.experimental
from .variable_selector import VariableSelector, PredicateSelector
from ..utils import broadcast_jaxtree

__all__ = [
    "ADVI",
    "guide",
    "Guide",
    "GuideProgram",
    "MeanfieldNormalGuide"
]

class GuideInitCtx(GuideContext):
    def __init__(self, rng_key: PRNGKey = jax.random.PRNGKey(0)) -> None:
        self.rng_key = rng_key
        self.X: Trace = dict()
        self.params: Dict[str,FloatArray] = dict()
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            raise Exception("Observations not supported for guides")
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        value = distribution.sample(sample_key)
        self.X[address] = value
        return value
    def param(self, address: str, init_value: FloatArrayLike, constraint: Constraint = real) -> FloatArrayLike:
        transform: transforms.Transform = transforms.biject_to(constraint)
        self.params[address] = transform.inv(init_value)
        return init_value


class GuideGenerateCtx(GuideContext):
    def __init__(self, rng_key: PRNGKey, params: Dict[str,FloatArray]) -> None:
        self.rng_key = rng_key
        self.X: Trace = dict()
        self.log_prob: FloatArray = jnp.array(0., float)
        self.params = params
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            raise Exception("Observations not supported for guides")
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        value = distribution.numpyro_base.rsample(sample_key)
        self.X[address] = value
        self.log_prob += distribution.log_prob(value)
        return value
    def param(self, address: str, init_value: FloatArrayLike, constraint: Constraint = real) -> FloatArrayLike:
        transform: transforms.Transform = transforms.biject_to(constraint)
        return transform(self.params[address])

class GuideLogprobCtx(GuideContext):
    def __init__(self, X: Trace, params: Dict[str,FloatArray]) -> None:
        self.X = X
        self.log_prob: FloatArray = jnp.array(0., float)
        self.params = params
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            raise Exception("Observations not supported for guides")
        value = cast(DIST_SUPPORT, self.X[address])
        self.log_prob += distribution.log_prob(value)
        return value
    def param(self, address: str, init_value: FloatArrayLike, constraint: Constraint = real) -> FloatArrayLike:
        transform: transforms.Transform = transforms.biject_to(constraint)
        return transform(self.params[address])

class Guide(ABC):
    @abstractmethod
    def get_params(self) -> jax.Array:
        raise NotImplementedError
    @abstractmethod
    def update_params(self, params: jax.Array):
        raise NotImplementedError
    @abstractmethod
    def sample_and_log_prob(self, rng_key: PRNGKey, shape = ()) -> Tuple[Trace, FloatArray]:
        raise NotImplementedError
    @abstractmethod
    def sample(self, rng_key: PRNGKey, shape = ()) -> Trace:
        raise NotImplementedError
    @abstractmethod
    def log_prob(self, X: Trace) -> FloatArray:
        raise NotImplementedError
    
class GuideProgram(Guide):
    def __init__(self, f: Callable, args, kwargs) -> None:
        self.f = f
        self.args = args
        self.kwargs = kwargs
        with GuideInitCtx() as ctx:
            f(*args, **kwargs)
            self.guide_trace = ctx.params
            self.param_arr, self.unravel_fn = jax.flatten_util.ravel_pytree(ctx.params)
    def get_params(self) -> FloatArray:
       return self.param_arr
    def update_params(self, params: FloatArray):
       self.param_arr = params
    def _sample_and_log_prob(self, rng_key: FloatArray) -> Tuple[Trace, FloatArray]:
       with GuideGenerateCtx(rng_key, self.unravel_fn(self.param_arr)) as ctx:
            self.f(*self.args, **self.kwargs)
            return ctx.X, ctx.log_prob
    def sample_and_log_prob(self, rng_key: FloatArray, shape = ()) -> Tuple[Trace, FloatArray]:
        if shape == ():
            return self._sample_and_log_prob(rng_key)
        else:
            flat_keys = jax.random.split(rng_key, (prod(shape),))
            X, lp = jax.vmap(self._sample_and_log_prob)(flat_keys)
            X = jax.tree.map(lambda v: v.reshape(shape + v.shape[1:]), X)
            lp = lp.reshape(shape)
            return X, lp
    def sample(self, rng_key: PRNGKey, shape = ()) -> Trace:
        return self.sample_and_log_prob(rng_key, shape)[0]
    def log_prob(self, X: Trace) -> FloatArray:
       with GuideLogprobCtx(X, self.unravel_fn(self.param_arr)) as ctx:
            self.f(*self.args, **self.kwargs)
            return ctx.log_prob

def guide(f:Callable) -> Callable[..., GuideProgram]:
    def _f(*args, **kwargs):
        return GuideProgram(f, args, kwargs)
    return _f

from dccxjax.distributions import Normal
class MeanfieldNormalGuide(Guide):
    def __init__(self, slp: SLP, variable_selector: VariableSelector, init_sigma: float = 1.) -> None:
        self._variable_selector = variable_selector
        X = {addr: val for addr, val in slp.decision_representative.items() if variable_selector.contains(addr)}
        self.Y = {addr: val for addr, val in slp.decision_representative.items() if not variable_selector.contains(addr)}
        is_discret_map = slp.get_is_discrete_map()
        assert all(not is_discret_map[addr] for addr in X.keys())
        flat_X, unravel_fn = jax.flatten_util.ravel_pytree(X)
        assert len(flat_X.shape) == 1
        self.n_latents = flat_X.shape[0]
        self.mu = jax.lax.zeros_like_array(flat_X)
        self.omega = jax.lax.full_like(flat_X, jnp.log(init_sigma))
        self.unravel_fn = unravel_fn
    def get_params(self) -> FloatArray:
       return jax.lax.concatenate((self.mu, self.omega), 0)
    def update_params(self, params: FloatArray):
       self.mu = params[:self.n_latents]
       self.omega = params[self.n_latents:]
    def sample_and_log_prob(self, rng_key: FloatArray, shape = ()) -> Tuple[Trace, FloatArray]:
        d = Normal(self.mu, jax.lax.exp(self.omega))
        x = d.numpyro_base.rsample(rng_key, shape)
        lp = d.log_prob(x).sum(axis=-1)
        # print(x.shape) # shape + (self.n_latents,)
        if shape == ():
            X = self.unravel_fn(x) | self.Y
        else:
            x_flat = x.reshape(-1, self.n_latents)
            X = jax.vmap(self.unravel_fn)(x_flat)
            X = jax.tree.map(lambda v: v.reshape(shape + v.shape[1:]), X) | broadcast_jaxtree(self.Y, shape)
        # jax.debug.print("x={x} m={m} s={s}", x=x, m=self.mu, s=self.omega)
        return X, lp
    def sample(self, rng_key: PRNGKey, shape = ()) -> Trace: 
        return self.sample_and_log_prob(rng_key, shape)[0] 
    def log_prob(self, X: Trace) -> FloatArray:
        d = Normal(self.mu, jax.lax.exp(self.omega))
        x, _ = jax.flatten_util.ravel_pytree(X)
        lp = d.log_prob(x)
        return lp
    def variable_selector(self) -> VariableSelector:
        return self._variable_selector
    

class ADVIState(NamedTuple, Generic[OPTIMIZER_STATE]):
    iteration: int
    optimizer_state: OPTIMIZER_STATE

def make_advi_step(slp: SLP, guide: Guide, optimizer: Optimizer[OPTIMIZER_STATE], L: int):
    # log_prob_fn = gibbs_model.tempered_log_prob(jnp.array(1.,float), {})
    log_prob_fn = slp.log_prob
    def elbo_fn(params: jax.Array, rng_key: PRNGKey) -> FloatArray:
        guide.update_params(params)
        if L == 1:
            X, lq = guide.sample_and_log_prob(rng_key)
            lp = log_prob_fn(X)
            # jax.debug.print("{X} {lp} {lq}", X=X, lp=lp, lq=lq)
            elbo = lp - lq
        else:
            def _elbo_step(elbo: FloatArray, sample_key: PRNGKey) -> Tuple[FloatArray, None]:
                X, lq = guide.sample_and_log_prob(sample_key)
                lp = log_prob_fn(X)
                return elbo + (lp - lq), None
            elbo, _ = jax.lax.scan(_elbo_step, jnp.array(0., float), jax.random.split(rng_key, L))
            elbo = elbo / L
        return elbo
    
    def advi_step(advi_state: ADVIState[OPTIMIZER_STATE], rng_key: PRNGKey) -> Tuple[ADVIState[OPTIMIZER_STATE], FloatArray]:
        iteration, optimizer_state = advi_state
        params = optimizer.get_params_fn(optimizer_state)
        elbo, elbo_grad = jax.value_and_grad(elbo_fn, argnums=0)(params, rng_key)
        # jax.debug.print("e={e} g={g}", e=elbo, g=elbo_grad)
        new_optimizer_state = optimizer.update_fn(iteration, -elbo_grad, optimizer_state)
        return ADVIState(iteration + 1, new_optimizer_state), elbo
    
    return advi_step

def get_advi_scan_with_progressbar(kernel: Callable[[ADVIState[OPTIMIZER_STATE],PRNGKey],Tuple[ADVIState[OPTIMIZER_STATE],FloatArray]], progressbar_mngr: ProgressbarManager, n_iter: int) -> Callable[[ADVIState[OPTIMIZER_STATE],PRNGKey],Tuple[ADVIState[OPTIMIZER_STATE],FloatArray]]:
    progressbar_mngr.set_num_samples(n_iter)
    kernel_with_bar = _add_progress_bar(kernel, progressbar_mngr, n_iter)
    def scan_with_bar(init: ADVIState[OPTIMIZER_STATE], xs: PRNGKey) -> Tuple[ADVIState[OPTIMIZER_STATE],FloatArray]:
        progressbar_mngr.start_progress()
        jax.experimental.io_callback(progressbar_mngr._init_tqdm, None, 0)
        return jax.lax.scan(kernel_with_bar, init, xs)
    return jax.jit(scan_with_bar)

def get_advi_scan_without_progressbar(kernel: Callable[[ADVIState[OPTIMIZER_STATE],PRNGKey],Tuple[ADVIState[OPTIMIZER_STATE],FloatArray]]) -> Callable[[ADVIState[OPTIMIZER_STATE],PRNGKey],Tuple[ADVIState[OPTIMIZER_STATE],FloatArray]]:
    def scan_without_bar(init: ADVIState[OPTIMIZER_STATE], xs: PRNGKey) -> Tuple[ADVIState[OPTIMIZER_STATE],FloatArray]:
        return jax.lax.scan(kernel, init, xs)
    return jax.jit(scan_without_bar)

class ADVI(Generic[OPTIMIZER_STATE]):
    def __init__(self,
                 slp: SLP,
                 guide: Guide,
                 optimizer: Optimizer[OPTIMIZER_STATE],
                 L: int,
                 *,
                 progress_bar: bool = False) -> None:
        self.slp = slp
        self.guide = guide
        self.optimizer = optimizer
        self.L = L
        self.progress_bar = progress_bar
        self.progressbar_mngr = ProgressbarManager("ADVI for "+self.slp.formatted())

        self.advi_step = make_advi_step(slp, guide, optimizer, L)

        self.cached_advi_scan: Optional[Callable[[ADVIState[OPTIMIZER_STATE],PRNGKey],Tuple[ADVIState[OPTIMIZER_STATE],FloatArray]]] = None

    def continue_run(self, rng_key: PRNGKey, state: ADVIState[OPTIMIZER_STATE], *, iteration: int = 0, n_iter: int):
        keys = jax.random.split(rng_key, n_iter)
        init_state = ADVIState(iteration, state.optimizer_state)

        if self.cached_advi_scan:
            self.progressbar_mngr.set_num_samples(n_iter)
            last_state, elbo = self.cached_advi_scan(init_state, keys)
        else:
            scan_fn = (
                get_advi_scan_with_progressbar(self.advi_step, self.progressbar_mngr, n_iter)
                if self.progress_bar else
                get_advi_scan_without_progressbar(self.advi_step)
            )
            self.cached_advi_scan = scan_fn
            last_state, elbo = scan_fn(init_state, keys)

        return last_state, elbo

    def run(self, rng_key: PRNGKey, *, n_iter: int):
        init_state = ADVIState(0, self.optimizer.init_fn(self.guide.get_params()))
        return self.continue_run(rng_key, init_state, n_iter=n_iter)
        
    def get_updated_guide(self, state: ADVIState[OPTIMIZER_STATE]) -> Guide:
        p = self.optimizer.get_params_fn(state.optimizer_state)
        self.guide.update_params(p)
        return self.guide
   
