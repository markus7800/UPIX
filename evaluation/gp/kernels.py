import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from dccxjax import FloatArrayLike, FloatArray
import dccxjax.distributions as dist

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
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
    @abstractmethod
    def __repr__(self, vals: bool = False) -> str:
        raise NotImplementedError
    
    def posterior_predictive(self, xs: FloatArray, ys: FloatArray, noise: FloatArrayLike, xs_pred: FloatArray, noise_pred: FloatArrayLike) -> dist.MultivariateNormal:
        n_prev = len(xs)
        n_new = len(xs_pred)
        means = jnp.zeros(n_prev + n_new, float)
        cov_matrix = self.eval_cov_vec(jnp.concatenate((xs, xs_pred)))
        cov_matrix_11 = cov_matrix[:n_prev,:n_prev] + (noise * jnp.eye(n_prev))
        cov_matrix_22 = cov_matrix[n_prev:,n_prev:]
        cov_matrix_12 = cov_matrix[:n_prev,n_prev:]
        cov_matrix_21 = cov_matrix[n_prev:,:n_prev]
        # print(f"{n_prev=}, {n_new=}, {cov_matrix_11.shape=} {cov_matrix_22.shape=} {cov_matrix_12.shape=} {cov_matrix_21.shape=}")

        mu1 = means[:n_prev]
        mu2 = means[n_prev:]

        conditional_mu = mu2 + cov_matrix_21 @ jnp.linalg.solve(cov_matrix_11, (ys - mu1))
        conditional_cov_matrix = cov_matrix_22 - cov_matrix_21 @ jnp.linalg.solve(cov_matrix_11, cov_matrix_12)
        conditional_cov_matrix = 0.5 * conditional_cov_matrix + 0.5 * jnp.transpose(conditional_cov_matrix)
        conditional_cov_matrix = conditional_cov_matrix + (noise_pred * jnp.eye(n_new))

        return dist.MultivariateNormal(conditional_mu, conditional_cov_matrix)

    
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
    def __repr__(self, vals: bool = False) -> str:
        if vals:
            return f"Const({self.value.item():.2f})"
        else:
            return "Const"
    def name(self) -> str:
        return "Constant"
 
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
    def __repr__(self, vals: bool = False) -> str:
        if vals:
            return f"Lin({self.intercept.item():.2f}; {self.bias.item():.2f}, {self.amplitude.item():.2f})"
        else:
            return "Lin"
    def name(self) -> str:
        return "Linear"
    
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
    def __repr__(self, vals: bool = False) -> str:
        if vals:
            return f"SqExp({self.lengthscale.item():.2f}; {self.amplitude.item():.2f})"
        else:
            return "SqExp"
    def name(self) -> str:
        return "SquaredExponential"

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
    def __repr__(self, vals: bool = False) -> str:
        if vals:
            return f"GamExp({self.lengthscale.item():.2f}, {self.gamma.item():.2f}; {self.amplitude.item():.2f})"
        else:
            return "GamExp"
    def name(self) -> str:
        return "GammaExponential"

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
    def __repr__(self, vals: bool = False) -> str:
        if vals:
            return f"Per({self.lengthscale.item():.2f}, {self.period.item():.2f}; {self.amplitude.item():.2f})"
        else:
            return "Per"
    def name(self) -> str:
        return "Periodic"

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
    def __repr__(self, vals: bool = False) -> str:
        return f"({self.left.__repr__(vals)} + {self.right.__repr__(vals)})"
    def name(self) -> str:
        return "Plus"
   
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
    def __repr__(self, vals: bool = False) -> str:
        return f"({self.left.__repr__(vals)} * {self.right.__repr__(vals)})"
    def name(self) -> str:
        return "Times"
    

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
        return transform_log_normal(z, -1.5, 1.) # if z is Normal(0,1) then param is LogNormal(-1.5, 1.)

def untransform_param(field: str, param: FloatArrayLike):
    if field == "gamma":
        return untransform_logit_normal(param, 2, 0, 1)
    else:
        return untransform_log_normal(param, -1.5, 1.)

# import numpyro.distributions as numpyro_dist
# import matplotlib.pyplot as plt
# z = jax.random.normal(jax.random.PRNGKey(0), (10_000_000,))
# param = transform_param("", z)
# param = param[param < 5]
# plt.hist(param, density=True, bins=100)
# xs = jnp.linspace(0, 5, 1000)
# ps = jnp.exp(numpyro_dist.LogNormal(-1.5, 1.0).log_prob(xs))
# plt.plot(xs, ps)
# plt.show()

