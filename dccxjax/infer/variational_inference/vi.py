import jax.flatten_util
from dccxjax.core.samplecontext import GuideContext
from dccxjax.distributions import Transform, Distribution, DIST_SUPPORT, DIST_SUPPORT_LIKE, TransformedDistribution, MultivariateNormal, Normal
from typing import Optional, Sequence, List, Dict, Tuple, NamedTuple, Callable, Generic, TypeVar, Any, cast
from dccxjax.types import FloatArray, FloatArrayLike, PRNGKey, Trace, IntArray
from abc import abstractmethod, ABC
from enum import Enum
import jax
import jax.numpy as jnp
from dccxjax.distributions.constraints import ConstraintT, real, scaled_unit_lower_cholesky
import numpyro.distributions.transforms as transforms
from dccxjax.core.model_slp import Model, SLP
from dccxjax.progress_bar import _add_progress_bar, ProgressbarManager
from dccxjax.infer.variational_inference.optimizers import OPTIMIZER_STATE, Optimizer
from math import prod
from dccxjax.infer.gibbs_model import GibbsModel
import jax.experimental
from dccxjax.infer.variable_selector import VariableSelector, PredicateSelector
from dccxjax.utils import broadcast_jaxtree
from dccxjax.jax_utils import smap_vmap
from dccxjax.parallelisation import ParallelisationConfig, SHARDING_AXIS, VectorisationType, vectorise_scan, parallel_run, batched_vmap
from tqdm.auto import tqdm

__all__ = [
    "ADVI",
    "guide",
    "Guide",
    "GuideProgram",
    "MeanfieldNormalGuide",
    "FullRankNormalGuide"
]

class GuideInitCtx(GuideContext):
    def __init__(self, rng_key: PRNGKey = jax.random.key(0)) -> None:
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
    def param(self, address: str, init_value: FloatArrayLike, constraint: ConstraintT = real) -> FloatArrayLike:
        transform: transforms.Transform = transforms.biject_to(constraint)
        self.params[address] = cast(FloatArray, transform.inv(init_value))
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
        value = cast(DIST_SUPPORT, distribution.numpyro_base.rsample(sample_key))
        self.X[address] = value
        self.log_prob += distribution.log_prob(value)
        return value
    def param(self, address: str, init_value: FloatArrayLike, constraint: ConstraintT = real) -> FloatArrayLike:
        transform: transforms.Transform = transforms.biject_to(constraint)
        return cast(FloatArrayLike,transform(self.params[address]))

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
    def param(self, address: str, init_value: FloatArrayLike, constraint: ConstraintT = real) -> FloatArrayLike:
        transform: transforms.Transform = transforms.biject_to(constraint)
        return cast(FloatArrayLike,transform(self.params[address]))

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

from dccxjax.distributions import Normal, MultivariateNormal
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
        self.mu = jax.numpy.zeros_like(flat_X)
        self.omega = jax.lax.full_like(flat_X, jnp.log(init_sigma))
        self.unravel_fn = unravel_fn
    def get_params(self) -> FloatArray:
       return jax.lax.concatenate((self.mu, self.omega), 0)
    def update_params(self, params: FloatArray):
       self.mu = params[:self.n_latents]
       self.omega = params[self.n_latents:]
    def sample_and_log_prob(self, rng_key: PRNGKey, shape = ()) -> Tuple[Trace, FloatArray]:
        d = Normal(self.mu, jax.lax.exp(self.omega))
        x = d.rsample(rng_key, shape)
        lp = d.log_prob(x).sum(axis=-1)
        # print(x.shape) # shape + (self.n_latents,)
        if shape == ():
            X = self.unravel_fn(x) | self.Y
        else:
            x_flat = x.reshape(-1, self.n_latents)
            X = jax.vmap(self.unravel_fn)(x_flat)
            X = jax.tree.map(lambda v: v.reshape(shape + v.shape[1:]), X) | broadcast_jaxtree(self.Y, shape)
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

# adapted from numpyro/infer/autoguide/guides.py#AutoMultivariateNormal

class FullRankNormalGuide(Guide):
    def __init__(self, slp: SLP, variable_selector: VariableSelector, init_sigma: float = 1.) -> None:
        self._variable_selector = variable_selector
        X = {addr: val for addr, val in slp.decision_representative.items() if variable_selector.contains(addr)}
        self.Y = {addr: val for addr, val in slp.decision_representative.items() if not variable_selector.contains(addr)}
        is_discret_map = slp.get_is_discrete_map()
        assert all(not is_discret_map[addr] for addr in X.keys())
        flat_X, unravel_fn = jax.flatten_util.ravel_pytree(X)
        assert len(flat_X.shape) == 1
        self.n_latents = flat_X.shape[0]
        self.mu = jax.numpy.zeros_like(flat_X)
        self.transform_to_cholesky: transforms.Transform = transforms.biject_to(scaled_unit_lower_cholesky)
        self.L = self.transform_to_cholesky.inv(jnp.eye(self.n_latents)*init_sigma)
        self.unravel_fn = unravel_fn
    def get_params(self) -> FloatArray:
       return jax.lax.concatenate((self.mu, self.L), 0)
    def update_params(self, params: FloatArray):
       self.mu = params[:self.n_latents]
       self.L = params[self.n_latents:]
    def sample_and_log_prob(self, rng_key: PRNGKey, shape = ()) -> Tuple[Trace, FloatArray]:
        scale_tril = cast(FloatArrayLike,self.transform_to_cholesky(self.L))
        d = MultivariateNormal(self.mu, scale_tril=scale_tril)
        x = d.rsample(rng_key, shape)
        lp = d.log_prob(x)
        if shape == ():
            X = self.unravel_fn(x) | self.Y
        else:
            x_flat = x.reshape(-1, self.n_latents)
            X = jax.vmap(self.unravel_fn)(x_flat)
            X = jax.tree.map(lambda v: v.reshape(shape + v.shape[1:]), X) | broadcast_jaxtree(self.Y, shape)
        return X, lp
    def sample(self, rng_key: PRNGKey, shape = ()) -> Trace: 
        return self.sample_and_log_prob(rng_key, shape)[0] 
    def log_prob(self, X: Trace) -> FloatArray:
        scale_tril = cast(FloatArrayLike,self.transform_to_cholesky(self.L))
        d = MultivariateNormal(self.mu, scale_tril=scale_tril)
        x, _ = jax.flatten_util.ravel_pytree(X)
        lp = d.log_prob(x)
        return lp
    def variable_selector(self) -> VariableSelector:
        return self._variable_selector
    

class ADVIState(NamedTuple, Generic[OPTIMIZER_STATE]):
    iteration: IntArray
    optimizer_state: OPTIMIZER_STATE

def make_advi_step(slp: SLP, guide: Guide, optimizer: Optimizer[OPTIMIZER_STATE], L: int, vectorisation: str, vmap_batch_size: int):
    assert vectorisation in ("vmap", "smap", "psum")
    # log_prob_fn = gibbs_model.tempered_log_prob(jnp.array(1.,float), {})
    log_prob_fn = slp.log_prob
    def elbo_fn(params: jax.Array, rng_key: PRNGKey) -> FloatArray:
        guide.update_params(params)
        if L == 1:
            X, lq = guide.sample_and_log_prob(rng_key)
            lp = log_prob_fn(X)
            elbo = lp - lq
        else:
            if vectorisation in "vmap":
                if vmap_batch_size > 0:
                    X, lq = batched_vmap(guide.sample_and_log_prob, batch_size=vmap_batch_size)(jax.random.split(rng_key, L))
                else:
                    X, lq = jax.vmap(guide.sample_and_log_prob)(jax.random.split(rng_key, L))
                lp = jax.vmap(log_prob_fn)(X)
                elbo = (lp - lq).sum() / L
            elif vectorisation in "smap":
                X, lq = smap_vmap(guide.sample_and_log_prob, axis_name=SHARDING_AXIS, in_axes=0, out_axes=0)(jax.random.split(rng_key, L))
                lp = smap_vmap(log_prob_fn, axis_name=SHARDING_AXIS, in_axes=0, out_axes=0)(X)
                elbo = (lp - lq).sum() / L
            else:
                assert vectorisation in "psum"
                X, lq = guide.sample_and_log_prob(rng_key)
                lp = log_prob_fn(X)
                elbo = lp - lq
                
            # def _elbo_step(elbo: FloatArray, sample_key: PRNGKey) -> Tuple[FloatArray, None]:
            #     X, lq = guide.sample_and_log_prob(sample_key)
            #     lp = log_prob_fn(X)
            #     return elbo + (lp - lq), None
            # elbo, _ = jax.lax.scan(_elbo_step, jnp.array(0., float), jax.random.split(rng_key, L))
            # elbo = elbo / L
        return elbo
    
    def advi_step(advi_state: ADVIState[OPTIMIZER_STATE], rng_key: PRNGKey) -> Tuple[ADVIState[OPTIMIZER_STATE], FloatArray]:
        iteration, optimizer_state = advi_state
        params = optimizer.get_params_fn(optimizer_state)
        elbo, elbo_grad = jax.value_and_grad(elbo_fn, argnums=0)(params, rng_key)
        if vectorisation in "psum":
            elbo = jax.lax.psum(elbo, SHARDING_AXIS) / L
            elbo_grad = jax.lax.psum(elbo_grad, SHARDING_AXIS) / L
        new_optimizer_state = optimizer.update_fn(cast(int, iteration), -elbo_grad, optimizer_state)
        return ADVIState(iteration + 1, new_optimizer_state), elbo
    
    return advi_step


class ADVI(Generic[OPTIMIZER_STATE]):
    def __init__(self,
                 slp: SLP,
                 guide: Guide,
                 optimizer: Optimizer[OPTIMIZER_STATE],
                 L: int,
                 *,
                 pconfig: ParallelisationConfig,
                 show_progress: bool = False,
                 shared_progressbar: tqdm | None = None) -> None:
        self.slp = slp
        self.guide = guide
        self.optimizer = optimizer
        self.L = L
        self.show_progress = show_progress
        self.progressbar_mngr = ProgressbarManager(
            "ADVI for "+self.slp.formatted(),
            shared_progressbar,
            thread_locked=pconfig.vectorisation==VectorisationType.PMAP
        )

        self.pconfig = pconfig
        
        if pconfig.vectorisation in (VectorisationType.GlobalSMAP, VectorisationType.PMAP):
            self.vectorisation = "psum"
            # params are replicated and gradients are shared across all jax devices
            # if device_count < L redundant computation happens as we only take one param set
            assert jax.device_count() >= self.L, f"L={self.L} cannot be greater than device_count={jax.device_count()} for {pconfig.vectorisation}.\nUse local smap instead or increase number of jax devices."
        elif pconfig.vectorisation == VectorisationType.LocalVMAP:
            self.vectorisation = "vmap"
        elif pconfig.vectorisation == VectorisationType.LocalSMAP:
            self.vectorisation = "smap"
        else:
            assert pconfig.vectorisation == VectorisationType.GlobalVMAP
            raise Exception(f"Vectoristiation: Global vmap not supported")

        self.advi_step = make_advi_step(slp, guide, optimizer, L, self.vectorisation, self.pconfig.vmap_batch_size)

        self.cached_advi_scan: Optional[Callable[[ADVIState[OPTIMIZER_STATE],PRNGKey],Tuple[ADVIState[OPTIMIZER_STATE],FloatArray]]] = None

    def continue_run(self, rng_key: PRNGKey, state: ADVIState[OPTIMIZER_STATE], *, iteration: IntArray = jnp.array(0,int), n_iter: int):
        
        self.progressbar_mngr.set_num_samples(n_iter)

        if self.vectorisation == "psum":
            keys = jax.random.split(rng_key, (n_iter,self.L))
            init_state = ADVIState(iteration, broadcast_jaxtree(state.optimizer_state, (self.L,)))
        else:
            keys = jax.random.split(rng_key, n_iter)
            init_state = ADVIState(iteration, state.optimizer_state)
            
        if self.cached_advi_scan:
            scan_fn = self.cached_advi_scan
        else:
            advi_state_axes = ADVIState(iteration=None, optimizer_state=0) # type: ignore
            scan_fn = vectorise_scan(self.advi_step,
                                     carry_axes=advi_state_axes,
                                     pmap_data_axes=1,
                                     batch_axis_size=self.L,
                                     vmap_batch_size=self.pconfig.vmap_batch_size,
                                     vectorisation=self.pconfig.vectorisation,
                                     progressbar_mngr=self.progressbar_mngr if self.show_progress else None,
                                     get_iternum_fn=lambda carry: carry.iteration)
            self.cached_advi_scan = scan_fn
        
        last_state, elbo = parallel_run(scan_fn, (init_state, keys), batch_axis_size=self.L, vectorisation=self.pconfig.vectorisation)

        return last_state, elbo

    def run(self, rng_key: PRNGKey, *, n_iter: int):
        init_state = ADVIState(jnp.array(0,int), self.optimizer.init_fn(self.guide.get_params()))
        return self.continue_run(rng_key, init_state, n_iter=n_iter)
        
    def get_updated_guide(self, state: ADVIState[OPTIMIZER_STATE]) -> Guide:
        p = self.optimizer.get_params_fn(state.optimizer_state)
        if self.vectorisation == "psum":
            p = p[0,...] # params are identical along batched axis
        self.guide.update_params(p)
        return self.guide
   
