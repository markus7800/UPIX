import jax
import jax.numpy as jnp
from evaluation.gp.data import get_data_autogp
import numpyro.distributions as dist
from evaluation.gp.kernels import *
from evaluation.gp.kernels import _gamma_exponential_cov, _rational_quadratic_cov

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
    cov_matrix = a * _gamma_exponential_cov(dt, l, g)
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



l = transform_param("lengthscale", -0.03259766)
s = transform_param("scale_mixture", -0.14109862)

def h(l, s):
    k = UnitRationalQuadratic(l, s)
    cov_matrix = k.eval_cov_vec(xs)
    return cov_matrix.sum()

print(h(l,s))

def h_star(x):
    return h(*x)

print(jax.grad(h_star)(jnp.array([l,s],float)))

def h2(l, s):
    dt = jax.lax.square((xs.reshape(-1,1) - xs.reshape(1,-1))/ l)
    cov_matrix = _rational_quadratic_cov(dt, s)
    return cov_matrix.sum()

print(h2(l,s))

def h2_star(x):
    return h2(*x)

print(jax.grad(h2_star)(jnp.array([l,s],float)))
