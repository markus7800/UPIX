from evaluation.gp.data import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from dccxjax import *
import dccxjax.distributions as dist
import numpyro.distributions as numpyro_dist
from evaluation.gp.kernels import *
from dataclasses import fields
from tqdm.auto import tqdm
from dccxjax.core.branching_tracer import retrace_branching
from time import time
from functools import reduce
from typing import Generic, NamedTuple
import jax.flatten_util

from dccxjax.infer.optimizers import OPTIMIZER_STATE, Optimizer, Adam

xs, xs_val, ys, ys_val = get_data_autogp()

def logdensity1(x):
    lengthscale = transform_param("lengthscale", x[0])
    scale_mixture = transform_param("scale_mixture", x[1])
    noise = transform_param("noise", x[2]) + 1e-5
    lp = jax.scipy.stats.norm.logpdf(lengthscale) + jax.scipy.stats.norm.logpdf(scale_mixture) + jax.scipy.stats.norm.logpdf(noise)
    k = UnitRationalQuadratic(lengthscale, scale_mixture)
    cov_matrix = k.eval_cov_vec(xs) + noise * jnp.eye(xs.size)
    return lp + jax.scipy.stats.multivariate_normal.logpdf(ys, jax.lax.zeros_like_array(xs), cov_matrix)

def logdensity2(x):
    lengthscale = transform_param("lengthscale", x[0])
    amplitude = transform_param("amplitude", x[1])
    noise = transform_param("noise", x[2]) + 1e-5
    lp = jax.scipy.stats.norm.logpdf(lengthscale) + jax.scipy.stats.norm.logpdf(amplitude) + jax.scipy.stats.norm.logpdf(noise)
    k = SquaredExponential(lengthscale, amplitude)
    cov_matrix = k.eval_cov_vec(xs) + noise * jnp.eye(xs.size)
    return lp + jax.scipy.stats.multivariate_normal.logpdf(ys, jax.lax.zeros_like_array(xs), cov_matrix)


class Meanfield:
    def __init__(self, X, init_sigma: float = 1.) -> None:
        flat_X, unravel_fn = jax.flatten_util.ravel_pytree(X)
        self.n_latents = flat_X.shape[0]
        self.mu = jax.lax.zeros_like_array(flat_X)
        self.omega = jax.lax.full_like(flat_X, jnp.log(init_sigma))
        print(flat_X, self.mu, self.omega)
        self.unravel_fn = unravel_fn
    def get_params(self) -> FloatArray:
       return jax.lax.concatenate((self.mu, self.omega), 0)
    def update_params(self, params: FloatArray):
       self.mu = params[:self.n_latents]
       self.omega = params[self.n_latents:]
    def sample_and_log_prob(self, rng_key: PRNGKey, shape = ()) -> Tuple[jax.Array, FloatArray]:
        d = numpyro_dist.Normal(self.mu, jax.lax.exp(self.omega))
        x = d.rsample(rng_key, shape)
        lp = d.log_prob(x).sum(axis=-1)
        # print(x.shape) # shape + (self.n_latents,)
        if shape == ():
            X = self.unravel_fn(x)
        else:
            x_flat = x.reshape(-1, self.n_latents)
            X = jax.vmap(self.unravel_fn)(x_flat)
            X = jax.tree.map(lambda v: v.reshape(shape + v.shape[1:]), X)
        return X, lp

class ADVIState(NamedTuple, Generic[OPTIMIZER_STATE]):
    iteration: int
    optimizer_state: OPTIMIZER_STATE

def make_advi_step(logdensity, guide: Meanfield, optimizer: Optimizer[OPTIMIZER_STATE], L: int):
    def elbo_fn(params: jax.Array, rng_key: PRNGKey) -> FloatArray:
        guide.update_params(params)
        if L == 1:
            X, lq = guide.sample_and_log_prob(rng_key)
            lp = logdensity(X)
            # jax.debug.print("{X} {lq} {lp}", X=X, lq=lq, lp=lp)
            elbo = lp - lq
        else:
            def _elbo_step(elbo: FloatArray, sample_key: PRNGKey) -> Tuple[FloatArray, None]:
                X, lq = guide.sample_and_log_prob(sample_key)
                lp = logdensity(X)
                return elbo + (lp - lq), None
            elbo, _ = jax.lax.scan(_elbo_step, jnp.array(0., float), jax.random.split(rng_key, L))
            elbo = elbo / L
        return elbo
    
    def advi_step(advi_state: ADVIState[OPTIMIZER_STATE], rng_key: PRNGKey) -> Tuple[ADVIState[OPTIMIZER_STATE], FloatArray]:
        iteration, optimizer_state = advi_state
        params = optimizer.get_params_fn(optimizer_state)
        elbo, elbo_grad = jax.value_and_grad(elbo_fn, argnums=0)(params, rng_key)
        new_optimizer_state = optimizer.update_fn(iteration, -elbo_grad, optimizer_state)
        return ADVIState(iteration + 1, new_optimizer_state), elbo
    
    return advi_step

optimizer = Adam(0.005)
X = (jnp.array(0.,float),jnp.array(0.,float),jnp.array(0.,float))
g = Meanfield(X, 0.1)
advi_step = make_advi_step(logdensity2, g, optimizer, 1)
keys = jax.random.split(jax.random.PRNGKey(0), 25_000)
t0 = time()
result, elbo = jax.lax.scan(advi_step, ADVIState(0, optimizer.init_fn(g.get_params())), keys)
print(elbo)
t1 = time()
print(f"Finished in {t1-t0:.3f}s")
# plt.plot(elbo)
# plt.show()