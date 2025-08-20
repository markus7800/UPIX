import jax
import jax.numpy as jnp
from dccxjax.distributions import Transform, Distribution, DIST_SUPPORT, DIST_SUPPORT_LIKE, TransformedDistribution, MultivariateNormal, Normal
import numpyro.distributions as numpyro_dists
from typing import Any, Optional, Dict, Callable, cast, Tuple
from abc import ABC, abstractmethod
from dccxjax.types import Trace, PRNGKey, FloatArrayLike, FloatArray, ArrayLike, BoolArray, IntArray
from dccxjax.distributions.constraints import Constraint, real

__all__ = [
    "sample",
    "logfactor",
    "factor",
    "param"
]

import threading
class SampleContextStore(threading.local):
    def __init__(self):
        self.ctx: Optional[SampleContext] = None
    def get(self):
        return self.ctx
    def set(self, ctx: Optional["SampleContext"]):
        self.ctx = ctx

SAMPLE_CONTEXT_STORE = SampleContextStore()

class SampleContext(ABC):
    @abstractmethod
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        raise NotImplementedError
    @abstractmethod
    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        raise NotImplementedError
    def __enter__(self):
        global SAMPLE_CONTEXT_STORE
        SAMPLE_CONTEXT_STORE.set(self)
        return self
    def __exit__(self, *args):
        global SAMPLE_CONTEXT_STORE
        SAMPLE_CONTEXT_STORE.set(None)

def sample(address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
    global SAMPLE_CONTEXT_STORE
    ctx = SAMPLE_CONTEXT_STORE.get()
    if ctx is not None:
        return ctx.sample(address, distribution, observed)
    else:
        raise Exception("Probabilistic program run without sample context")
    
def logfactor(f: FloatArrayLike, address: str = "__log_factor__") -> None:
    global SAMPLE_CONTEXT_STORE
    ctx = SAMPLE_CONTEXT_STORE.get()
    if ctx is not None:
        return ctx.logfactor(f, address)
    else:
        raise Exception("Probabilistic program run without sample context")

def factor(f: FloatArrayLike) -> None:
    logfactor(jax.lax.log(f))
    

class GenerateCtx(SampleContext):
    def __init__(self, rng_key: PRNGKey, Y: Trace = dict()) -> None:
        super().__init__()
        self.X: Trace = dict()
        self.Y: Trace = Y
        self.rng_key = rng_key
        self.log_likelihood: FloatArray = jnp.array(0.,float)
        self.log_prior: FloatArray = jnp.array(0.,float)
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            self.log_likelihood += distribution.log_prob(observed).sum()
            return cast(DIST_SUPPORT, observed)
        self.rng_key, key = jax.random.split(self.rng_key)
        if address not in self.Y:
            value = distribution.sample(key)
        else:
            value = cast(DIST_SUPPORT, self.Y[address])
        self.X[address] = value
        self.log_prior += distribution.log_prob(value).sum()
        return value
    def logfactor(self, lf: FloatArrayLike, address) -> None:
        self.log_likelihood += lf
        

AnnealingMask = Dict[str,BoolArray]

def maybe_annealed_log_prob(address:str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], value: DIST_SUPPORT, annealing_masks: AnnealingMask):
    if address in annealing_masks:
        mask = annealing_masks[address]
        if isinstance(distribution, MultivariateNormal) and len(value.shape) == 1:
            assert isinstance(distribution.numpyro_base, numpyro_dists.MultivariateNormal)
            cov_matrix_full: jax.Array = distribution.numpyro_base.covariance_matrix # type: ignore
            cov_matrix_masked = jax.lax.select(mask.reshape(1,-1) & mask.reshape(-1,1), cov_matrix_full, jnp.eye(value.size))
            lp = MultivariateNormal(covariance_matrix=cov_matrix_masked).log_prob(value)
            lp -= jax.lax.select(mask, jax.lax.zeros_like_array(value), Normal(0.,1.).log_prob(value)).sum()
            return lp
        else:
            lp = distribution.log_prob(value)
            assert distribution.numpyro_base.event_dim == 0 and len(lp.shape) == 1, f"Data-annealing only supported for univariate distributions, got {lp.shape=}"
            return jax.lax.select(mask, lp, jax.lax.zeros_like_array(lp)).sum()
    else:
        return distribution.log_prob(value).sum()


class LogprobCtx(SampleContext):
    def __init__(self, X: Trace, annealing_masks: AnnealingMask = dict()) -> None:
        super().__init__()
        self.X = X
        self.log_likelihood: FloatArray = jnp.array(0.,float)
        self.log_prior: FloatArray = jnp.array(0.,float)
        self.annealing_masks: AnnealingMask = annealing_masks
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            _observed = cast(DIST_SUPPORT, observed)
            self.log_likelihood += maybe_annealed_log_prob(address, distribution, _observed, self.annealing_masks)
            return _observed
        assert distribution.numpyro_base._validate_args
        value = cast(DIST_SUPPORT, self.X[address])

        self.log_prior +=  maybe_annealed_log_prob(address, distribution, value, self.annealing_masks)

        return value
    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        self.log_likelihood += lf

class LogprobTraceCtx(SampleContext):
    def __init__(self, X: Trace, annealing_masks: AnnealingMask = dict()) -> None:
        super().__init__()
        self.X = X
        self.log_probs: Dict[str,Tuple[FloatArray,bool]] = dict()
        self.annealing_masks: AnnealingMask = annealing_masks
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            _observed = cast(DIST_SUPPORT, observed)
            self.log_probs[address] = (maybe_annealed_log_prob(address, distribution, _observed, self.annealing_masks), True)
            return _observed
        assert distribution.numpyro_base._validate_args
        value = cast(DIST_SUPPORT, self.X[address])
        self.log_probs[address] = (maybe_annealed_log_prob(address, distribution, value, self.annealing_masks), False)
        return value
    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        self.log_probs[address] = (self.log_probs.get(address, jnp.array(0.,float))[0] + lf, True)
    

class ReplayCtx(SampleContext):
    def __init__(self, X: Trace) -> None:
        super().__init__()
        self.X = X
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            return cast(DIST_SUPPORT, observed)
        value = cast(DIST_SUPPORT, self.X[address])
        return value
    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        pass
    
class CollectDistributionTypesCtx(SampleContext):
    def __init__(self, X: Trace) -> None:
        super().__init__()
        self.X = X
        self.is_discrete: Dict[str, bool] = dict()
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            return cast(DIST_SUPPORT, observed)
        value = cast(DIST_SUPPORT,self.X[address])
        self.is_discrete[address] = distribution.numpyro_base.is_discrete
        return value
    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        pass


class UnconstrainedLogprobCtx(SampleContext):
    def __init__(self, X_unconstrained: Trace, annealing_masks: AnnealingMask = dict()) -> None:
        super().__init__()
        self.X_unconstrained = X_unconstrained
        self.X_constrained: Trace = dict()
        self.log_prior: FloatArray = jnp.array(0.,float)
        self.log_likelihood: FloatArray = jnp.array(0.,float)
        self.annealing_masks: AnnealingMask = annealing_masks
    
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            _observed = cast(DIST_SUPPORT, observed)
            self.log_likelihood += maybe_annealed_log_prob(address, distribution, _observed, self.annealing_masks)
            return _observed
        assert distribution.numpyro_base._validate_args

        unconstrained_value: FloatArray = self.X_unconstrained[address]
        
        if distribution.numpyro_base.is_discrete:
            constrained_value = cast(DIST_SUPPORT, unconstrained_value)
            self.X_constrained[address] = constrained_value
            self.log_prior += maybe_annealed_log_prob(address, distribution, constrained_value, self.annealing_masks)

        else:
            transform: Transform[FloatArray, DIST_SUPPORT] = distribution.biject_so_support()
            constrained_value = cast(DIST_SUPPORT, transform(unconstrained_value))
            self.X_constrained[address] = constrained_value

            unconstrained_distribution = TransformedDistribution(distribution, transform.inv())
            self.log_prior += unconstrained_distribution.log_prob(unconstrained_value).sum()
            maybe_annealed_log_prob(address, unconstrained_distribution, unconstrained_value, self.annealing_masks)
        return constrained_value
    
    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        self.log_likelihood += lf
    

class TransformToUnconstrainedCtx(SampleContext):
    def __init__(self, X_constrained: Trace) -> None:
        super().__init__()
        self.X_unconstrained: Trace = dict()
        self.X_constrained: Trace = X_constrained
    
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            return cast(DIST_SUPPORT, observed)
        assert distribution.numpyro_base._validate_args
        constrained_value = cast(DIST_SUPPORT, self.X_constrained[address])

        if distribution.numpyro_base.is_discrete:
            self.X_unconstrained[address] = constrained_value
        else:
            transform_inv: Transform[DIST_SUPPORT, FloatArray] = distribution.biject_so_support().inv()
            unconstrained_value = transform_inv(constrained_value)
            self.X_unconstrained[address] = unconstrained_value

        return constrained_value
    
    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        pass
    
class TransformToConstrainedCtx(SampleContext):
    def __init__(self, X_unconstrained: Trace) -> None:
        super().__init__()
        self.X_unconstrained: Trace = X_unconstrained
        self.X_constrained: Trace = dict()
    
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            return cast(DIST_SUPPORT, observed)
        assert distribution.numpyro_base._validate_args
        unconstrained_value: FloatArray = self.X_unconstrained[address]

        if distribution.numpyro_base.is_discrete:
            constrained_value = cast(DIST_SUPPORT, unconstrained_value)
            self.X_constrained[address] = constrained_value
        else:
            transform: Transform[FloatArray, DIST_SUPPORT] = distribution.biject_so_support()
            constrained_value = cast(DIST_SUPPORT, transform(unconstrained_value))
            self.X_constrained[address] = constrained_value

        return constrained_value
    
    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        pass
    

class GuideContext(SampleContext, ABC):
    @abstractmethod
    def param(self, address: str, init_value: FloatArrayLike, constraint: Constraint = real) -> FloatArrayLike:
        raise NotImplementedError

    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        raise Exception("logfactor not supported for guides")
    
def param(address: str, init_value: FloatArrayLike, constraint: Constraint = real) -> FloatArrayLike:
    global SAMPLE_CONTEXT_STORE
    ctx = SAMPLE_CONTEXT_STORE.get()
    if ctx is not None:
        assert isinstance(ctx, GuideContext)
        return ctx.param(address, init_value, constraint)
    else:
        raise Exception("Probabilistic program run without guide context")