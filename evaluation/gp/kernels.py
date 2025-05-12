import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from dccxjax import FloatArrayLike

class GPKernel(ABC):
    @abstractmethod
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        raise NotImplementedError
    @abstractmethod
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        raise NotImplementedError
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError
    @abstractmethod
    def depth(self) -> int:
        raise NotImplementedError
    
class PrimitiveGPKernel(GPKernel):
    def size(self) -> int:
        return 1
    def depth(self) -> int:
        return 1
class CompositiveGPKernel(GPKernel):
    pass

@dataclass
class Constant(PrimitiveGPKernel):
    value: jax.Array
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        return (t1 == t2) * self.value
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        return (ts.reshape(-1,1) == ts.reshape(1,-1)) * self.value
 
@dataclass
class Linear(PrimitiveGPKernel):
    intercept: jax.Array
    bias: jax.Array
    amplitude: jax.Array
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        c = (t1 - self.intercept) * (t2 - self.intercept)
        return self.bias + self.amplitude * c
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        ts = ts - self.intercept
        c = (ts.reshape(-1,1) * ts.reshape(1,-1))
        return self.bias + self.amplitude * c
    
@dataclass
class SquaredExponential(PrimitiveGPKernel):
    lengthscale: jax.Array
    amplitude: jax.Array
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        c = jax.lax.exp(-0.5 * (t1 - t2) * (t1 - t2) / jax.lax.square(self.lengthscale))
        return self.amplitude * c
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        dt = ts.reshape(-1,1) - ts.reshape(1,-1)
        c = jax.lax.exp(-0.5 * dt * dt / jax.lax.square(self.lengthscale))
        return self.amplitude * c

@dataclass
class GammaExponential(PrimitiveGPKernel):
    lengthscale: jax.Array
    gamma: jax.Array # in [0,2]
    amplitude: jax.Array
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        dt = jax.lax.abs(t1 - t2)
        c = jax.lax.exp(- (dt/self.lengthscale)**self.gamma)
        return self.amplitude * c
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        dt = jax.lax.abs(ts.reshape(-1,1) - ts.reshape(1,-1))
        c = jax.lax.exp(- (dt/self.lengthscale)**self.gamma)
        return self.amplitude * c

@dataclass
class Periodic(PrimitiveGPKernel):
    lengthscale: jax.Array
    period: jax.Array
    amplitude: jax.Array
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        freq = jnp.pi / self.period
        dt = jax.lax.abs(t1 - t2)
        c = jax.lax.exp((-2/jax.lax.square(self.lengthscale)) * jax.lax.square(jax.lax.sin(freq * dt)))
        return self.amplitude * c
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        freq = jnp.pi / self.period
        dt = jax.lax.abs(ts.reshape(-1,1) - ts.reshape(1,-1))
        c = jax.lax.exp((-2/jax.lax.square(self.lengthscale)) * jax.lax.square(jax.lax.sin(freq * dt)))
        return self.amplitude * c

@dataclass
class Plus(CompositiveGPKernel):
    left: GPKernel
    right: GPKernel
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        return self.left.eval_cov(t1, t2) + self.right.eval_cov(t1, t2)
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        return self.left.eval_cov_vec(ts) + self.right.eval_cov_vec(ts)
    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()
    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())
   
@dataclass 
class Times(CompositiveGPKernel):
    left: GPKernel
    right: GPKernel
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        return self.left.eval_cov(t1, t2) * self.right.eval_cov(t1, t2)
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        return self.left.eval_cov_vec(ts) * self.right.eval_cov_vec(ts)
    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()
    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())
    


def transform_log_normal(z: FloatArrayLike, mu: FloatArrayLike, sigma: FloatArrayLike):
    return jax.lax.exp(mu + sigma*z)
def untransform_log_normal(param: FloatArrayLike, mu: FloatArrayLike, sigma: FloatArrayLike):
    return (jax.lax.log(param) - mu) / sigma

def transform_logit_normal(z: FloatArrayLike, scale: FloatArrayLike, mu: FloatArrayLike, sigma: FloatArrayLike):
    return scale * 1 / (1 + jax.lax.exp(-(mu + sigma*z)))
def untransform_logit_normal(param: FloatArrayLike, scale: FloatArrayLike, mu: FloatArrayLike, sigma: FloatArrayLike):
    return (jax.lax.log(param / (scale - param)) - mu) / sigma

def transform_param(field: str, z: FloatArrayLike):
    if field == "gamma":
        return transform_logit_normal(z, 2, 0, 1)
    else:
        return transform_log_normal(z, -1.5, 1.)
    
def untransform_param(field: str, param: FloatArrayLike):
    if field == "gamma":
        return untransform_logit_normal(param, 2, 0, 1)
    else:
        return untransform_log_normal(param, -1.5, 1.)
    