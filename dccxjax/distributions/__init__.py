
import numpyro.distributions as numpyro_dists
import numpyro.distributions.transforms as numypro_transforms
from typing import Any, Generic, TypeVar, Optional
import jax
from ..types import *

numpyro_dists.Distribution.set_default_validate_args(True)

TRANSFORM_DOMAIN = TypeVar("TRANSFORM_DOMAIN", bound=jax.Array)
TRANSFORM_CODOMAIN = TypeVar("TRANSFORM_CODOMAIN", bound=jax.Array)
class Transform(Generic[TRANSFORM_DOMAIN, TRANSFORM_CODOMAIN]):
    def __init__(self, numpyro_transform: numypro_transforms.Transform) -> None:
        self.numpyro_transform = numpyro_transform
    def __call__(self, x: TRANSFORM_DOMAIN) -> TRANSFORM_CODOMAIN:
        return self.numpyro_transform(x)
    def inv(self) -> "Transform[TRANSFORM_CODOMAIN, TRANSFORM_DOMAIN]":
        return Transform(self.numpyro_transform.inv)

DIST_SUPPORT = TypeVar("DIST_SUPPORT", bound=jax.Array)
DIST_SUPPORT_LIKE = TypeVar("DIST_SUPPORT_LIKE", bound=ArrayLike)
class Distribution(Generic[DIST_SUPPORT,DIST_SUPPORT_LIKE]):
    def __init__(self, numpyro_base: numpyro_dists.Distribution) -> None:
        self.numpyro_base = numpyro_base
    def sample(self, key: PRNGKey, sample_shape=()) -> DIST_SUPPORT:
        return self.numpyro_base.sample(key, sample_shape=sample_shape) # type: ignore
    def log_prob(self, value: DIST_SUPPORT | DIST_SUPPORT_LIKE) -> FloatArray:
        return self.numpyro_base.log_prob(value)
    def biject_so_support(self) -> Transform[FloatArray, DIST_SUPPORT]:
        return Transform(numpyro_dists.biject_to(self.numpyro_base.support))


class Normal(Distribution[FloatArray,FloatArrayLike]):
    def __init__(self, loc: FloatArrayLike, scale: FloatArrayLike) -> None:
        super().__init__(numpyro_dists.Normal(loc, scale)) # type: ignore


class MultivariateNormal(Distribution[FloatArray,FloatArrayLike]):
    def __init__(self, loc: FloatArrayLike = 0, covariance_matrix: Optional[FloatArrayLike] = None, precision_matrix: Optional[FloatArrayLike] = None, scale_tril: Optional[FloatArrayLike] = None) -> None:
        super().__init__(numpyro_dists.MultivariateNormal(loc, covariance_matrix, precision_matrix, scale_tril)) # type: ignore
        
class Uniform(Distribution[FloatArray,FloatArrayLike]):
    def __init__(self, low: FloatArrayLike = 0., high: FloatArrayLike = 1.):
        super().__init__(numpyro_dists.Uniform(low, high)) # type: ignore

class Gamma(Distribution[FloatArray,FloatArrayLike]):
    def __init__(self, concentration: FloatArrayLike, rate: FloatArrayLike = 1.):
        super().__init__(numpyro_dists.Gamma(concentration, rate)) # type: ignore
        
class InverseGamma(Distribution[FloatArray,FloatArrayLike]):
    def __init__(self, concentration: FloatArrayLike, rate: FloatArrayLike = 1.):
        super().__init__(numpyro_dists.InverseGamma(concentration, rate)) # type: ignore

SimplexArray = jax.Array

class Dirichlet(Distribution[SimplexArray,SimplexArray]):
    def __init__(self, concentration: FloatArrayLike):
        super().__init__(numpyro_dists.Dirichlet(concentration)) # type: ignore

class Bernoulli(Distribution[IntArray,IntArrayLike]):
    def __init__(self, probs: FloatArrayLike):
        super().__init__(numpyro_dists.BernoulliProbs(probs)) # type: ignore

class Poisson(Distribution[IntArray,IntArrayLike]):
    def __init__(self, rate: FloatArrayLike):
        super().__init__(numpyro_dists.Poisson(rate)) # type: ignore

class Categorical(Distribution[IntArray,IntArrayLike]):
    def __init__(self, probs: SimplexArray):
        super().__init__(numpyro_dists.CategoricalProbs(probs)) # type: ignore

class CategoricalLogits(Distribution[IntArray,IntArrayLike]):
    def __init__(self, logits: FloatArrayLike):
        super().__init__(numpyro_dists.CategoricalLogits(logits)) # type: ignore

class TwoSidedTruncatedDistribution(Distribution[DIST_SUPPORT,DIST_SUPPORT_LIKE]):
    def __init__(self, base: Distribution[DIST_SUPPORT,DIST_SUPPORT_LIKE], low: FloatArrayLike = 0, high: FloatArrayLike = 1) -> None:
        super().__init__(numpyro_dists.TwoSidedTruncatedDistribution(base.numpyro_base, low, high))  # type: ignore

class TransformedDistribution(Distribution[TRANSFORM_CODOMAIN,TRANSFORM_CODOMAIN]):
    def __init__(self, base: Distribution[DIST_SUPPORT,DIST_SUPPORT_LIKE], transforms: Transform[DIST_SUPPORT, TRANSFORM_CODOMAIN]) -> None:
        super().__init__(numpyro_dists.TransformedDistribution(base.numpyro_base, transforms.numpyro_transform))