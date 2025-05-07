import jax
import jax.numpy as jnp
from dccxjax.distributions import Distribution, DIST_SUPPORT, DIST_SUPPORT_LIKE
import numpyro.distributions as numpyro_dists
from typing import Any, Optional, Dict, Callable, cast
from abc import ABC, abstractmethod
from ..types import Trace, PRNGKey, FloatArrayLike, FloatArray, ArrayLike

__all__ = [
    "sample",
    "logfactor",
    "factor"
]

class SampleContext(ABC):
    @abstractmethod
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        raise NotImplementedError
    @abstractmethod
    def logfactor(self, lf: FloatArrayLike) -> None:
        raise NotImplementedError
    def __enter__(self):
        global SAMPLE_CONTEXT
        SAMPLE_CONTEXT = self
        return self
    def __exit__(self, *args):
        global SAMPLE_CONTEXT
        SAMPLE_CONTEXT = None

SAMPLE_CONTEXT: Optional[SampleContext] = None
def sample(address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
    global SAMPLE_CONTEXT
    if SAMPLE_CONTEXT is not None:
        return SAMPLE_CONTEXT.sample(address, distribution, observed)
    else:
        raise Exception("Probabilistic program run without sample context")
    
def logfactor(f: FloatArrayLike) -> None:
    global SAMPLE_CONTEXT
    if SAMPLE_CONTEXT is not None:
        return SAMPLE_CONTEXT.logfactor(f)
    else:
        raise Exception("Probabilistic program run without sample context")

def factor(f: FloatArrayLike) -> None:
    logfactor(jax.lax.log(f))
    

class GenerateCtx(SampleContext):
    def __init__(self, rng_key: PRNGKey, Y: Trace = dict()) -> None:
        super().__init__()
        self.X: Trace = dict() | Y
        self.rng_key = rng_key
        self.log_likelihood: FloatArray = jnp.array(0.,float)
        self.log_prior: FloatArray = jnp.array(0.,float)
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            self.log_likelihood += distribution.log_prob(observed).sum()
            return cast(DIST_SUPPORT, observed)
        self.rng_key, key = jax.random.split(self.rng_key)
        if address not in self.X:
            value = distribution.sample(key)
            self.X[address] = value
        else:
            value = cast(DIST_SUPPORT, self.X[address])
        self.log_prior += distribution.log_prob(value).sum()
        return value
    def logfactor(self, lf: FloatArrayLike) -> None:
        self.log_likelihood += lf
        
class LogprobCtx(SampleContext):
    def __init__(self, X: Trace) -> None:
        super().__init__()
        self.X = X
        self.log_likelihood: FloatArray = jnp.array(0.,float)
        self.log_prior: FloatArray = jnp.array(0.,float)
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            self.log_likelihood += distribution.log_prob(observed).sum()
            return cast(DIST_SUPPORT, observed)
        assert distribution.numpyro_base._validate_args
        value = cast(DIST_SUPPORT, self.X[address])
        self.log_prior += distribution.log_prob(value).sum()
        return value
    def logfactor(self, lf: FloatArrayLike) -> None:
        self.log_likelihood += lf
    

class ReplayCtx(SampleContext):
    def __init__(self, X: Trace) -> None:
        super().__init__()
        self.X = X
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            return cast(DIST_SUPPORT, observed)
        value = cast(DIST_SUPPORT, self.X[address])
        return value
    def logfactor(self, lf: FloatArrayLike) -> None:
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
    def logfactor(self, lf: FloatArrayLike) -> None:
        pass


class UnconstrainedLogprobCtx(SampleContext):
    def __init__(self, X_unconstrained: Trace) -> None:
        super().__init__()
        self.X_unconstrained = X_unconstrained
        self.X_constrained: Trace = dict()
        self.log_prob: FloatArray = jnp.array(0.,float)
    
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            self.log_prob += distribution.log_prob(observed).sum()
            return cast(DIST_SUPPORT, observed)
        assert distribution.numpyro_base._validate_args

        unconstrained_value = self.X_unconstrained[address]
        
        if distribution.numpyro_base.is_discrete:
            constrained_value = cast(DIST_SUPPORT, unconstrained_value)
            self.X_constrained[address] = constrained_value
        else:
            transform = numpyro_dists.biject_to(distribution.numpyro_base.support)
            constrained_value = cast(DIST_SUPPORT, transform(unconstrained_value))
            self.X_constrained[address] = constrained_value

            unconstrained_distribution = numpyro_dists.TransformedDistribution(distribution, transform.inv)
            
            self.log_prob += unconstrained_distribution.log_prob(unconstrained_value).sum()

        return constrained_value
    
    def logfactor(self, lf: FloatArrayLike) -> None:
        self.log_prob += lf
    

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
            transform = numpyro_dists.biject_to(distribution.numpyro_base.support)
            unconstrained_value = transform.inv(constrained_value)
            self.X_unconstrained[address] = unconstrained_value

        return constrained_value
    
    def logfactor(self, lf: FloatArrayLike) -> None:
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
        unconstrained_value = self.X_unconstrained[address]

        if distribution.numpyro_base.is_discrete:
            constrained_value = cast(DIST_SUPPORT, unconstrained_value)
            self.X_constrained[address] = constrained_value
        else:
            transform = numpyro_dists.biject_to(distribution.numpyro_base.support)
            constrained_value = cast(DIST_SUPPORT, transform(unconstrained_value))
            self.X_constrained[address] = constrained_value

        return constrained_value
    
    def logfactor(self, lf: FloatArrayLike) -> None:
        pass