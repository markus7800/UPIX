import jax.flatten_util
from ..core.samplecontext import SampleContext, SAMPLE_CONTEXT
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

__all__ = [
    "param",
    "SGD",
    "Momentum",
    "Adagrad",
    "Adam",
    "advi",
    "guide"
]


class GuideContext(SampleContext, ABC):
    @abstractmethod
    def param(self, address: str, init_value: FloatArrayLike, constraint: Constraint = real) -> FloatArrayLike:
        raise NotImplementedError

    def logfactor(self, lf: FloatArrayLike) -> None:
        raise Exception("logfactor not supported for guides")
    
def param(address: str, init_value: FloatArrayLike, constraint: Constraint = real) -> FloatArrayLike:
    global SAMPLE_CONTEXT
    if SAMPLE_CONTEXT is not None:
        assert isinstance(SAMPLE_CONTEXT, GuideContext)
        return SAMPLE_CONTEXT.param(address, init_value, constraint)
    else:
        raise Exception("Probabilistic program run without guide context")

class GuideInitCtx(GuideContext):
    def __init__(self, rng_key: PRNGKey = jax.random.PRNGKey(0)) -> None:
        self.rng_key = rng_key
        self.X: Trace = dict()
        self.params: Dict[str,FloatArray] = dict()
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is None:
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
        if observed is None:
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
        if observed is None:
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
    def sample_and_log_prob(self, rng_key: PRNGKey) -> Tuple[Trace, FloatArray]:
        raise NotImplementedError
    @abstractmethod
    def sample(self, rng_key: PRNGKey) -> Trace:
        raise NotImplementedError
    @abstractmethod
    def log_prob(self, X: Trace) -> FloatArray:
        raise NotImplementedError
class GuideModel(Guide):
    def __init__(self, f: Callable, args, kwargs) -> None:
        self.f = f
        self.args = args
        self.kwargs = kwargs
        with GuideInitCtx() as ctx:
            f(*args, **kwargs)
            self.param_arr, self.unravel_fn = jax.flatten_util.ravel_pytree(ctx.params)
    def get_params(self) -> FloatArray:
       return self.param_arr
    def update_params(self, params: FloatArray):
       self.param_arr = params
    def sample_and_log_prob(self, rng_key: FloatArray) -> Tuple[Trace, FloatArray]:
       with GuideGenerateCtx(rng_key, self.unravel_fn(self.param_arr)) as ctx:
            self.f(*self.args, **self.kwargs)
            return ctx.X, ctx.log_prob
    def sample(self, rng_key: PRNGKey) -> Trace:
        return self.sample_and_log_prob(rng_key)[0]
    def log_prob(self, X: Trace) -> FloatArray:
       with GuideLogprobCtx(X, self.unravel_fn(self.param_arr)) as ctx:
            self.f(*self.args, **self.kwargs)
            return ctx.log_prob

def guide(f:Callable) -> Callable[..., GuideModel]:
    def _f(*args, **kwargs):
        return GuideModel(f, args, kwargs)
    return _f


# we do not need pytrees / we operate on arrays
OPTIMIZER_STATE = TypeVar("OPTIMIZER_STATE")
OPTIMIZER_PARAMS = jax.Array
OPTIMIZER_UPDATES = jax.Array
class Optimizer(NamedTuple, Generic[OPTIMIZER_STATE]):
    init_fn: Callable[[OPTIMIZER_PARAMS], OPTIMIZER_STATE]
    update_fn: Callable[[int, OPTIMIZER_UPDATES, OPTIMIZER_STATE], OPTIMIZER_STATE]
    get_params_fn: Callable[[OPTIMIZER_STATE], OPTIMIZER_PARAMS]

def SGD(step_size: float) -> Optimizer:
  def init(x0: jax.Array) -> jax.Array:
    return x0
  def update(i: int, g: jax.Array, x: jax.Array) -> jax.Array:
    return x - step_size * g
  def get_params(x):
    return x
  return Optimizer(init, update, get_params)

def Momentum(step_size: float, mass: float):
  def init(x0: jax.Array) -> Tuple[jax.Array,jax.Array]:
    v0 = jnp.zeros_like(x0)
    return x0, v0
  def update(i: int, g: jax.Array, state: Tuple[jax.Array,jax.Array]) -> Tuple[jax.Array,jax.Array]:
    x, velocity = state
    velocity = mass * velocity + g
    x = x - step_size * velocity
    return x, velocity
  def get_params(state: Tuple[jax.Array,jax.Array]) -> jax.Array:
    x, _ = state
    return x
  return Optimizer(init, update, get_params)

def Adagrad(step_size: float, momentum=0.9) -> Optimizer:
  def init(x0: jax.Array) -> Tuple[jax.Array,jax.Array,jax.Array]:
    g_sq = jnp.zeros_like(x0)
    m = jnp.zeros_like(x0)
    return x0, g_sq, m

  def update(i: int, g: jax.Array, state: Tuple[jax.Array,jax.Array,jax.Array]) -> Tuple[jax.Array,jax.Array,jax.Array]:
    x, g_sq, m = state
    g_sq += jnp.square(g)
    g_sq_inv_sqrt = jnp.where(g_sq > 0, 1. / jnp.sqrt(g_sq), 0.0)
    m = (1. - momentum) * (g * g_sq_inv_sqrt) + momentum * m
    x = x - step_size * m
    return x, g_sq, m

  def get_params(state: Tuple[jax.Array,jax.Array,jax.Array]) -> jax.Array:
    x, _, _ = state
    return x

  return Optimizer(init, update, get_params)

def Adam(step_size: float, b1=0.9, b2=0.999, eps=1e-8) -> Optimizer:
  def init(x0: jax.Array) -> Tuple[jax.Array,jax.Array,jax.Array]:
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    return x0, m0, v0
  def update(i: int, g: jax.Array, state: Tuple[jax.Array,jax.Array,jax.Array]):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
    mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
    vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
    x = x - step_size * mhat / (jnp.sqrt(vhat) + eps)
    return x, m, v
  def get_params(state: Tuple[jax.Array,jax.Array,jax.Array]) -> jax.Array:
    x, _, _ = state
    return x
  return Optimizer(init, update, get_params)

class ADVIState(NamedTuple, Generic[OPTIMIZER_STATE]):
    iteration: int
    optimizer_state: OPTIMIZER_STATE

def make_advi_step(slp: SLP, guide: Guide, optimizer: Optimizer[OPTIMIZER_STATE], L: int):
    def elbo_fn(params: jax.Array, rng_key: PRNGKey) -> FloatArray:
        guide.update_params(params)
        if L == 1:
            X, lq = guide.sample_and_log_prob(rng_key)
            lp = slp.log_prob(X)
            elbo = lp - lq
        else:
            def _elbo_step(elbo: FloatArray, sample_key: PRNGKey) -> Tuple[FloatArray, None]:
                X, lq = guide.sample_and_log_prob(sample_key)
                lp = slp.log_prob(X)
                return elbo + (lp - lq), None

            elbo, _ = jax.lax.scan(_elbo_step, jnp.array(0., float), jax.random.split(rng_key, L))
            elbo = elbo / L
        return elbo
    
    def advi_step(advi_state: ADVIState[OPTIMIZER_STATE], rng_key: PRNGKey) -> Tuple[ADVIState[OPTIMIZER_STATE], FloatArray]:
        iteration, optimizer_state = advi_state
        params = optimizer.get_params_fn(optimizer_state)
        elbo, elbo_grad = jax.value_and_grad(elbo_fn, argnums=0)(params, rng_key)
        new_optimizer_state = optimizer.update_fn(iteration, -elbo_grad, optimizer_state)
        return ADVIState(iteration + 1, new_optimizer_state), elbo
    
    return advi_step

import jax.experimental
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
            self.cached_smc_scan = scan_fn
            last_state, elbo = scan_fn(init_state, keys)

        return last_state, elbo

    def run(self, rng_key: PRNGKey, *, n_iter: int):
        init_state = ADVIState(0, self.optimizer.init_fn(self.guide.get_params()))
        return self.continue_run(rng_key, init_state, n_iter=n_iter)
        
        
   
