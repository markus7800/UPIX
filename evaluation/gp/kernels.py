import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, field
from dccxjax.types import FloatArrayLike, FloatArray
import dccxjax.distributions as dist
from typing import List

@dataclass
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
    def n_internal(self) -> int:
        return (self.size() - 1) // 2
    def n_leaf(self) -> int:
        return self.n_internal() + 1
    @abstractmethod
    def depth(self) -> int:
        raise NotImplementedError
    @staticmethod
    def name() -> str:
        raise NotImplementedError
    def __repr__(self) -> str:
        raise NotImplementedError
    @abstractmethod
    def key(self) -> str:
        raise NotImplementedError
    @abstractmethod
    def pprint(self) -> str:
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
    

def mvnormal_quantiles(distribution: dist.MultivariateNormal, quantiles: List[float]):
    broadcasted_normal = dist.Normal(distribution.mean, jnp.sqrt(distribution.variance))
    return [broadcasted_normal.numpyro_base.icdf(q) for q in quantiles]
def mvnormal_quantile(distribution: dist.MultivariateNormal, quantile: float):
    return mvnormal_quantiles(distribution, [quantile])[0]

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
    def __repr__(self) -> str:
        return self.key()
    def pprint(self) -> str:
            return f"Const({self.value.item():.2f})"
    def key(self) -> str:
            return "Const"
    @staticmethod
    def name() -> str:
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
    def __repr__(self) -> str:
        return self.key()
    def pprint(self) -> str:
            return f"Lin({self.intercept.item():.2f}; {self.bias.item():.2f}, {self.amplitude.item():.2f})"
    def key(self) -> str:
            return "Lin"
    @staticmethod
    def name() -> str:
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
    def __repr__(self) -> str:
        return self.key()
    def pprint(self) -> str:
        return f"SqExp({self.lengthscale.item():.2f}; {self.amplitude.item():.2f})"
    def key(self) -> str:
        return "SqExp"
    @staticmethod
    def name() -> str:
        return "SquaredExponential"

@jax.custom_jvp
def _gamma_exponential_cov(dt, l, g):
    return jax.lax.exp(-(dt/l)**g)
_gamma_exponential_cov.defjvps(
    lambda dt_dot, primal_out, dt, l, g: -primal_out/dt * dt_dot,
    lambda l_dot, primal_out, dt, l, g: primal_out * (g/l) * (dt/l)**g * l_dot,
    lambda g_dot, primal_out, dt, l, g: jnp.where(dt == 0., 0., -primal_out * (dt/l)**g * jax.lax.log(dt/l)) * g_dot,
)

@dataclass
class GammaExponential(PrimitiveGPKernel):
    lengthscale: jax.Array
    gamma: jax.Array # in [0,2]
    amplitude: jax.Array
    
    # def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
    #     dt = jax.lax.abs(t1 - t2)
    #     c = jax.lax.exp(- (dt/self.lengthscale)**self.gamma)
    #     return self.amplitude * c
    # def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
    #     dt = jax.lax.abs(ts.reshape(-1,1) - ts.reshape(1,-1))
    #     c = jax.lax.exp(- (dt/self.lengthscale)**self.gamma)
    #     return self.amplitude * c
    
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        dt = jax.lax.abs(t1 - t2)
        return self.amplitude * _gamma_exponential_cov(dt, self.lengthscale, self.gamma)
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        dt = jax.lax.abs(ts.reshape(-1,1) - ts.reshape(1,-1))
        return self.amplitude * _gamma_exponential_cov(dt, self.lengthscale, self.gamma)
    
    def __repr__(self) -> str:
        return self.key()
    def pprint(self) -> str:
        return f"GamExp({self.lengthscale.item():.2f}, {self.gamma.item():.2f}; {self.amplitude.item():.2f})"
    def key(self) -> str:
        return "GamExp"
    @staticmethod
    def name() -> str:
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
    def __repr__(self) -> str:
        return self.key()
    def pprint(self) -> str:
        return f"Per({self.lengthscale.item():.2f}, {self.period.item():.2f}; {self.amplitude.item():.2f})"
    def key(self) -> str:
        return "Per"
    @staticmethod
    def name() -> str:
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
    def __repr__(self) -> str:
        return f"({self.left.__repr__()} + {self.right.__repr__()})"
    def pprint(self) -> str:
        return f"({self.left.pprint()} + {self.right.pprint()})"
    def key(self) -> str:
        return f"({self.left.key()} + {self.right.key()})"
    @staticmethod
    def name() -> str:
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
    def __repr__(self) -> str:
        return f"({self.left.__repr__()} * {self.right.__repr__()})"
    def pprint(self) -> str:
        return f"({self.left.pprint()} * {self.right.pprint()})"
    def key(self) -> str:
        return f"({self.left.key()} * {self.right.key()})"
    @staticmethod
    def name() -> str:
        return "Times"
    

def transform_log_normal(z: FloatArrayLike, mu: FloatArrayLike, sigma: FloatArrayLike):
    return jax.lax.exp(mu + sigma*z)
def transform_logit_normal(z: FloatArrayLike, scale: FloatArrayLike, mu: FloatArrayLike, sigma: FloatArrayLike):
    return scale * 1 / (1 + jax.lax.exp(-(mu + sigma*z)))
def transform_param(field: str, z: FloatArrayLike):
    if field == "gamma":
        return transform_logit_normal(z, 2, 0, 1)
    else:
        return transform_log_normal(z, -1.5, 1.) # if z is Normal(0,1) then param is LogNormal(-1.5, 1.)

def untransform_logit_normal(param: FloatArrayLike, scale: FloatArrayLike, mu: FloatArrayLike, sigma: FloatArrayLike):
    return (jax.lax.log(param / (scale - param)) - mu) / sigma
def untransform_log_normal(param: FloatArrayLike, mu: FloatArrayLike, sigma: FloatArrayLike):
    return (jax.lax.log(param) - mu) / sigma
def untransform_param(field: str, param: FloatArrayLike):
    if field == "gamma":
        return untransform_logit_normal(param, 2, 0, 1)
    else:
        return untransform_log_normal(param, -1.5, 1.)


@jax.custom_jvp
def _rational_quadratic_cov(dt, s):
    return (1 + (0.5/s) * dt) ** (-s)
_rational_quadratic_cov.defjvps(
    lambda dt_dot, primal_out, dt, s: -0.5 * primal_out/(1 + 0.5/s * dt) * dt_dot,
    lambda s_dot, primal_out, dt, s: (primal_out * ((-s - 0.5*dt) * jax.lax.log(1 + 0.5/s * dt) + 0.5*dt) / (s + 0.5*dt)) * s_dot
)

@dataclass
class RationalQuadratic(PrimitiveGPKernel):
    lengthscale: jax.Array
    scale_mixture: jax.Array
    amplitude: jax.Array
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        dt = jax.lax.square((t1 - t2)/self.lengthscale)
        # return self.amplitude * (1 + (0.5/self.scale_mixture) * dt) ** (-self.scale_mixture)
        return self.amplitude * _rational_quadratic_cov(dt, self.scale_mixture)
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        dt = jax.lax.square((ts.reshape(-1,1) - ts.reshape(1,-1)) / self.lengthscale)
        # return self.amplitude * (1 + (0.5/self.scale_mixture) * dt) ** (-self.scale_mixture)
        return self.amplitude * _rational_quadratic_cov(dt, self.scale_mixture)
    def __repr__(self) -> str:
        return self.key()
    def pprint(self) -> str:
        return f"RQ({self.lengthscale.item():.2f}, {self.scale_mixture.item():.2f}; {self.amplitude.item():.2f})"
    def key(self) -> str:
        return "RQ"
    @staticmethod
    def name() -> str:
        return "RationalQuadratic"

# == SquaredExponential
# @dataclass
# class RadialBasisFunction(PrimitiveGPKernel):
#     lengthscale: jax.Array
#     amplitude: jax.Array
    
#     def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
#         dt = jax.lax.square((t1 - t2)/self.lengthscale)
#         return self.amplitude * jax.lax.exp(-0.5 * dt)
#     def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
#         dt = jax.lax.square((ts.reshape(-1,1) - ts.reshape(1,-1)) / self.lengthscale)
#         return self.amplitude * jax.lax.exp(-0.5 * dt)
#     def __repr__(self) -> str:
#         return self.key()
#     def pprint(self) -> str:
#         return f"RBF({self.lengthscale.item():.2f}; {self.amplitude.item():.2f})"
#     def key(self) -> str:
#         return "RBF"
#     @staticmethod
#     def name() -> str:
#         return "RadialBasisFunction"

@dataclass
class Polynomial(PrimitiveGPKernel):
    bias: jax.Array
    amplitude: jax.Array
    degree: int
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        dt = t1 * t2 # univariate dot product
        return self.amplitude * (self.bias + dt)**self.degree
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        dt = ts.reshape(-1,1) * ts.reshape(1,-1)  # univariate dot product
        return self.amplitude * (self.bias + dt)**self.degree
    def __repr__(self) -> str:
        return self.key()
    def pprint(self) -> str:
        return f"Poly({self.bias.item():.2f}, {self.degree}; {self.amplitude.item():.2f})"
    def key(self) -> str:
        return "Poly"
    @staticmethod
    def name() -> str:
        return "Polynomial"


@dataclass
class UnitRationalQuadratic(PrimitiveGPKernel):
    lengthscale: jax.Array
    scale_mixture: jax.Array
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        return RationalQuadratic(self.lengthscale, self.scale_mixture, jnp.array(1.,float)).eval_cov(t1,t2)
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        return RationalQuadratic(self.lengthscale, self.scale_mixture, jnp.array(1.,float)).eval_cov_vec(ts)
    def __repr__(self) -> str:
        return self.key()
    def pprint(self) -> str:
        return f"1RQ({self.lengthscale.item():.2f}, {self.scale_mixture.item():.2f})"
    def key(self) -> str:
        return "1RQ"
    @staticmethod
    def name() -> str:
        return "UnitRationalQuadratic"
    
@dataclass
class UnitPolynomialDegreeOne(PrimitiveGPKernel):
    bias: jax.Array
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        return Polynomial(self.bias, jnp.array(1.,float), 1).eval_cov(t1,t2)
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        return Polynomial(self.bias, jnp.array(1.,float), 1).eval_cov_vec(ts)
    def __repr__(self) -> str:
        return self.key()
    def pprint(self) -> str:
        return f"1Poly({self.bias.item():.2f})"
    def key(self) -> str:
        return "1Poly"
    @staticmethod
    def name() -> str:
        return "UnitPolynomialDegreeOne"
    
@dataclass
class UnitPeriodic(PrimitiveGPKernel):
    lengthscale: jax.Array
    period: jax.Array
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        return Periodic(self.lengthscale, self.period, jnp.array(1.,float)).eval_cov(t1,t2)
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        return Periodic(self.lengthscale, self.period, jnp.array(1.,float)).eval_cov_vec(ts)
    def __repr__(self) -> str:
        return self.key()
    def pprint(self) -> str:
        return f"1Per({self.lengthscale.item():.2f}, {self.period.item():.2f})"
    def key(self) -> str:
        return "1Per"
    @staticmethod
    def name() -> str:
        return "UnitPeriodic"
    
@dataclass
class UnitSquaredExponential(PrimitiveGPKernel):
    lengthscale: jax.Array
    def eval_cov(self, t1: jax.Array, t2: jax.Array) -> jax.Array:
        return SquaredExponential(self.lengthscale, jnp.array(1.,float)).eval_cov(t1,t2)
    def eval_cov_vec(self, ts: jax.Array) -> jax.Array:
        return SquaredExponential(self.lengthscale, jnp.array(1.,float)).eval_cov_vec(ts)
    def __repr__(self) -> str:
        return self.key()
    def pprint(self) -> str:
        return f"1SqExp({self.lengthscale.item():.2f})"
    def key(self) -> str:
        return "1SqExp"
    @staticmethod
    def name() -> str:
        return "UnitSquaredExponential"
    
    
# import numpyro.distributions as numpyro_dist
# import matplotlib.pyplot as plt
# z = jax.random.normal(jax.random.key(0), (10_000_000,))
# param = transform_param("", z)
# param = param[param < 5]
# plt.hist(param, density=True, bins=100)
# xs = jnp.linspace(0, 5, 1000)
# ps = jnp.exp(numpyro_dist.LogNormal(-1.5, 1.0).log_prob(xs))
# plt.plot(xs, ps)
# plt.show()

