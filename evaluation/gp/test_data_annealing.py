import sys
sys.path.insert(0, ".")

from data import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from dccxjax.all import *
import dccxjax.distributions as dist
import numpyro.distributions as numpyro_dist
from kernels import *
from dataclasses import fields
from tqdm.auto import tqdm
from dccxjax.core.branching_tracer import retrace_branching
from time import time
from data import get_data_autogp

import logging
setup_logging(logging.WARN)

def get_kernel(X: Trace):
    lengthscale = transform_param("lengthscale", X["lengthscale"])
    period = transform_param("period", X["period"])
    amplitude = transform_param("amplitude", X["amplitude"])
    noise = transform_param("noise", X["noise"] if "noise" in X else 0.) 
    k = Periodic(lengthscale, period, amplitude)
    return k, noise


k, noise = get_kernel({"lengthscale": jnp.array(0.,float), "period": jnp.array(0.,float), "amplitude": jnp.array(0.,float), "noise": jnp.array(0.,float)})

xs, xs_val, ys, ys_val = get_data_autogp()

mask = jax.random.bernoulli(jax.random.key(0), 0.5, xs.shape).astype(bool)

cov_matrix_full = k.eval_cov_vec(xs) + (noise + 1e-5) * jnp.eye(xs.size)
print(f"{cov_matrix_full.shape}")

cov_matrix_masked = k.eval_cov_vec(xs[mask]) + (noise + 1e-5) * jnp.eye(xs[mask].size)
print(f"{cov_matrix_masked.shape}")

lp1 = dist.MultivariateNormal(covariance_matrix=cov_matrix_masked).log_prob(ys[mask])
print(lp1)

print(cov_matrix_full)

cov_matrix_selected = cov_matrix_full[mask.reshape(1,-1) & mask.reshape(-1,1)].reshape(xs[mask].size, xs[mask].size)
print(f"{cov_matrix_selected.shape}")

ixs = jnp.arange(0,xs.size)
cov_matrix_selected_2 = cov_matrix_full[jnp.ix_(ixs[mask],ixs[mask])]
print(f"{cov_matrix_selected_2.shape}")
print(jnp.sum(jnp.abs(cov_matrix_selected_2 - cov_matrix_selected)))


lp2 = dist.MultivariateNormal(covariance_matrix=cov_matrix_selected).log_prob(ys[mask])
print(lp2)

cov_matrix_masked = jax.lax.select(mask.reshape(1,-1) & mask.reshape(-1,1), cov_matrix_full, jnp.eye(xs.size))
print(f"{cov_matrix_masked.shape}")

lp3 = dist.MultivariateNormal(covariance_matrix=cov_matrix_masked).log_prob(ys)
lp3 -= jax.lax.select(mask, jax.lax.zeros_like_array(ys), dist.Normal(0.,1.).log_prob(ys)).sum()
print(lp3)