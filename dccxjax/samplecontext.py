import jax
import numpyro.distributions as dist
from typing import Any, Optional, Dict, Callable
from abc import ABC, abstractmethod
import jax._src.core as jax_core
from .types import Trace, PRNGKey
from .utils import maybe_jit_warning

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
        self.X: Dict[str, jax.Array] = dict()
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
        self.log_prior += distribution.log_prob(value)
        return value
    

class LogprobCtx(SampleContext):
    def __init__(self, X: Dict[str, jax.Array]) -> None:
        super().__init__()
        self.X = X
        self.log_prob = 0
    def sample(self, address: str, distribution: dist.Distribution, observed: Optional[jax.Array] = None) -> jax.Array:
        if observed is not None:
            self.log_prob += distribution.log_prob(observed).sum()
            return observed
        value = self.X[address]
        self.log_prob += distribution.log_prob(value)
        return value
    

class Model:
    def __init__(self, f: Callable, args, kwargs) -> None:
        self.f = f
        self.args = args
        self.kwargs = kwargs

        self._jitted_log_prob = False
        # self._log_prob = self.make_model_logprob()

    def __call__(self) -> Any:
        return self.f(*self.args, **self.kwargs)
    
    # not jitted
    def log_prob(self, X: Trace) -> float:
        with LogprobCtx(X) as ctx:
            self.f(*self.args, **self.kwargs)
            return ctx.log_prob

    def __repr__(self) -> str:
        return f"Model({self.f.__name__}, {self.args}, {self.kwargs})"
    
    def short_repr(self) -> str:
        return f"Model({self.f.__name__} at {hex(id(self))})"

def model(f):
    def _f(*args, **kwargs):
        return Model(f, args, kwargs)
    return _f