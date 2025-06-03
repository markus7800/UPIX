import jax
import jax.numpy as jnp
from evaluation.gp.data import get_data_autogp
import numpyro.distributions as dist
from evaluation.gp.kernels import *

xs, xs_val, ys, ys_val = get_data_autogp()

a = transform_param("amplitude", 0.00370225)
g = transform_param("gamma", -0.14109862)
l = transform_param("lengthscale", -0.03259766)

# verified for g == 2. and g==1.
# a = 1.5
# g = 2.
# l = 0.5

def f(a, g, l):
    k = GammaExponential(l, g, a)
    cov_matrix = k.eval_cov_vec(xs)
    return cov_matrix.sum()

print(f(a, g, l))

def f_star(x):
    return f(*x)

print(jax.grad(f_star)(jnp.array([a,g,l],float))) # nan at lengthscale


def f2(a, g, l):
    dt = jax.lax.abs(xs.reshape(-1,1) - xs.reshape(1,-1))
    cov_matrix = gamma_exponential_cov(dt, l, g, a)
    return cov_matrix.sum()

print(f2(a, g, l))

def f2_star(x):
    return f2(*x)

print(jax.grad(f2_star)(jnp.array([a,g,l],float))) # nan at lengthscale

dt = jax.lax.abs(xs.reshape(-1,1) - xs.reshape(1,-1))

grad_a = jax.lax.exp(- (dt/l)**g).sum()
print(grad_a)
grad_g = jnp.where(dt == 0., 0., -a * jax.lax.exp(- (dt/l)**g) * (dt/l)**g * jax.lax.log(dt/l)).sum()
print(grad_g)
grad_l = (a * g * jax.lax.exp(- (dt/l)**g) * (dt/l)**g / l).sum()
print(grad_l)
