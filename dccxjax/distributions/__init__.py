
import numpyro.distributions as numypro_dists
from typing import Generic, TypeVar
import jax
from ..types import *

numypro_dists.Distribution.set_default_validate_args(True)

DIST_SUPPORT = TypeVar("DIST_SUPPORT", bound=jax.Array)
DIST_SUPPORT_LIKE = TypeVar("DIST_SUPPORT_LIKE", bound=ArrayLike)
class Distribution(Generic[DIST_SUPPORT,DIST_SUPPORT_LIKE]):
    base: numypro_dists.Distribution
    def __init__(self, base: numypro_dists.Distribution) -> None:
        self.numpyro_base = base
    def sample(self, key: PRNGKey, sample_shape=()) -> DIST_SUPPORT:
        return self.numpyro_base.sample(key, sample_shape=sample_shape)
    def log_prob(self, value: DIST_SUPPORT | DIST_SUPPORT_LIKE) -> FloatArray:
        return self.numpyro_base.log_prob(value)


class Normal(Distribution[FloatArray,FloatArrayLike]):
    def __init__(self, loc: FloatArrayLike, scale: FloatArrayLike) -> None:
        super().__init__(numypro_dists.Normal(loc, scale)) # type: ignore
        
class Uniform(Distribution[FloatArray,FloatArrayLike]):
    def __init__(self, low: FloatArrayLike = 0., high: FloatArrayLike = 1.):
        super().__init__(numypro_dists.Uniform(low, high)) # type: ignore

class Bernoulli(Distribution[IntArray,IntArrayLike]):
    def __init__(self, probs: FloatArrayLike):
        super().__init__(numypro_dists.BernoulliProbs(probs)) # type: ignore