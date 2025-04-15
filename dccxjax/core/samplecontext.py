import jax
import dccxjax.distributions as dist
from typing import Any, Optional, Dict, Callable
from abc import ABC, abstractmethod
from ..types import Trace, PRNGKey


__all__ = [
    "sample",
]

class SampleContext(ABC):
    @abstractmethod
    def sample(self, address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
        raise NotImplementedError
    def __enter__(self):
        global SAMPLE_CONTEXT
        SAMPLE_CONTEXT = self
        return self
    def __exit__(self, *args):
        global SAMPLE_CONTEXT
        SAMPLE_CONTEXT = None

SAMPLE_CONTEXT: Optional[SampleContext] = None
def sample(address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
    global SAMPLE_CONTEXT
    if SAMPLE_CONTEXT is not None:
        return SAMPLE_CONTEXT.sample(address, distribution, observed)
    else:
        raise Exception("Probabilistic program run without sample context")
    

class GenerateCtx(SampleContext):
    def __init__(self, rng_key: PRNGKey) -> None:
        super().__init__()
        self.X: Trace = dict()
        self.rng_key = rng_key
        self.log_likelihood = 0.
        self.log_prior = 0.
    def sample(self, address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
        if observed is not None:
            self.log_likelihood += distribution.log_prob(observed).sum()
            return observed
        self.rng_key, key = jax.random.split(self.rng_key)
        value = distribution.sample(key)
        self.X[address] = value
        self.log_prior += distribution.log_prob(value).sum()
        return value
    

class LogprobCtx(SampleContext):
    def __init__(self, X: Trace) -> None:
        super().__init__()
        self.X = X
        self.log_likelihood = 0.
        self.log_prior = 0.
    def sample(self, address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
        if observed is not None:
            self.log_likelihood += distribution.log_prob(observed).sum()
            return observed
        assert distribution._validate_args
        value = self.X[address]
        self.log_prior += distribution.log_prob(value).sum()
        return value
    

class ReplayCtx(SampleContext):
    def __init__(self, X: Trace) -> None:
        super().__init__()
        self.X = X
    def sample(self, address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
        if observed is not None:
            return observed
        value = self.X[address]
        return value
    
class CollectDistributionTypesCtx(SampleContext):
    def __init__(self, X: Trace) -> None:
        super().__init__()
        self.X = X
        self.is_discrete: Dict[str, bool] = dict()
        
    def sample(self, address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
        if observed is not None:
            return observed
        value = self.X[address]
        self.is_discrete[address] = distribution.is_discrete
        return value


class UnconstrainedLogprobCtx(SampleContext):
    def __init__(self, X_unconstrained: Trace) -> None:
        super().__init__()
        self.X_unconstrained = X_unconstrained
        self.X_constrained: Trace = dict()
        self.log_prob = 0.
    
    def sample(self, address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
        if observed is not None:
            self.log_prob += distribution.log_prob(observed).sum()
            return observed
        assert distribution._validate_args

        unconstrained_value = self.X_unconstrained[address]
        
        if distribution.is_discrete:
            constrained_value = unconstrained_value
            self.X_constrained[address] = constrained_value
        else:
            transform = dist.biject_to(distribution.support)
            constrained_value = transform(unconstrained_value)
            self.X_constrained[address] = constrained_value

            unconstrained_distribution = dist.TransformedDistribution(distribution, transform.inv)
            
            self.log_prob += unconstrained_distribution.log_prob(unconstrained_value).sum()

        return constrained_value
    

class TransformToUnconstrainedCtx(SampleContext):
    def __init__(self, X_constrained: Trace) -> None:
        super().__init__()
        self.X_unconstrained: Trace = dict()
        self.X_constrained: Trace = X_constrained
    
    def sample(self, address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
        if observed is not None:
            return observed
        assert distribution._validate_args
        constrained_value = self.X_constrained[address]

        if distribution.is_discrete:
            self.X_unconstrained[address] = constrained_value
        else:
            transform = dist.biject_to(distribution.support)
            unconstrained_value = transform.inv(constrained_value)
            self.X_unconstrained[address] = unconstrained_value

        return constrained_value
    
class TransformToConstrainedCtx(SampleContext):
    def __init__(self, X_unconstrained: Trace) -> None:
        super().__init__()
        self.X_unconstrained: Trace = X_unconstrained
        self.X_constrained: Trace = dict()
    
    def sample(self, address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
        if observed is not None:
            return observed
        assert distribution._validate_args
        unconstrained_value = self.X_unconstrained[address]

        if distribution.is_discrete:
            constrained_value = unconstrained_value
            self.X_constrained[address] = constrained_value
        else:
            transform = dist.biject_to(distribution.support)
            constrained_value = transform(unconstrained_value)
            self.X_constrained[address] = constrained_value

        return constrained_value