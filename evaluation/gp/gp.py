import sys
sys.path.insert(0, ".")

from data import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from dccxjax import *
import dccxjax.distributions as dist
from kernels import *
from dataclasses import fields
from tqdm.auto import tqdm

xs, xs_val, ys, ys_val = get_data_autogp()

# plt.scatter(xs, ys)
# plt.scatter(xs_val, ys_val)
# plt.show()

def covariance_prior(idx: int) -> GPKernel:
    node_type = sample(f"{idx}_node_type", dist.Categorical(jnp.array([0.0, 0.21428571428571427, 0.0, 0.21428571428571427, 0.21428571428571427, 0.17857142857142858, 0.17857142857142858])))
    print(f"{idx}_node_type", node_type)
    if node_type < 5:
        NodeType = [Constant, Linear, SquaredExponential, GammaExponential, Periodic][node_type]
        params = []
        for field in fields(NodeType):
            field_name = field.name
            log_param = sample(f"{idx}_{field_name}", dist.Normal(0., 1.))
            param = transform_param(field_name, log_param)
            params.append(param)
        return NodeType(*params)
    else:
        NodeType = [Plus, Times][node_type - 5]
        left = covariance_prior(2*idx)
        right = covariance_prior(2*idx+1)
        return NodeType(left, right)
    
@model
def gaussian_process(ts: jax.Array):
    kernel = covariance_prior(1)
    noise = sample("noices", dist.Normal(0.,1.))
    noise = transform_param("noise", noise) + 1e-5
    cov_matrix = kernel.eval_cov_vec(ts) + noise * jnp.eye(ts.size)
    # MultivariateNormal does cholesky internally
    sample("obs", dist.MultivariateNormal(covariance_matrix=cov_matrix), observed=ts)


m = gaussian_process(ys)

rng_key = jax.random.PRNGKey(0)

rng_key, key = jax.random.split(rng_key)
rng_key, key = jax.random.split(rng_key)
X = sample_from_prior(m, key)
print(X)
exit()

active_slps: list[SLP] = []
for _ in tqdm(range(1_000)):
    rng_key, key = jax.random.split(rng_key)
    X = sample_from_prior(m, key)
    print(X)

    if all(slp.path_indicator(X) == 0 for slp in active_slps):
        slp = slp_from_decision_representative(m, X)
        active_slps.append(slp)

print(f"{len(active_slps)=}")