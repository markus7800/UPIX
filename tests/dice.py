from upix.core import *
import upix.distributions as dist
import jax
import jax.numpy as jnp
from upix.infer.exact import *
from typing import List

def diamond_rec(s1, N: int):
    route = sample(f"route_{N}", dist.Bernoulli(0.5))
    s2 = jax.lax.select(route, s1, 0)
    s3 = jax.lax.select(route, 0, s1)
    drop = sample(f"drop_{N}", dist.Bernoulli(0.001))
    res = s2 | (s3 & (1-drop))
    # res = sample(f"res_{N}", dist.Dirac(s2 | (s3 & (1-drop))))
    if N == 0:
        return res
    else:
        return diamond_rec(res, N-1)

# @model
# def diamond(N):
#     s1 = jnp.array(1,int)
#     for i in range(N):
#         route = sample(f"route_{i}", dist.Bernoulli(0.5))
#         s2 = jax.lax.select(route, s1, 0)
#         s3 = jax.lax.select(route, 0, s1)
#         drop = sample(f"drop_{i}", dist.Bernoulli(0.001))
#         s1 = sample(f"res_{i}", dist.Dirac(s2 | (s3 & (1-drop))))
#     return s1


@model
def diamond(N: int):
    diamond_rec(jnp.array(1,int), N)
    
    
slp = SLP_from_branchless_model(diamond(2))
    
supports = get_supports(slp)
print(supports)

fs = compute_factors(slp, supports, True)
print(fs)

marginal_variables = ["res_2"]
elimination_order = get_greedy_elimination_order(fs, marginal_variables)

@jax.jit
def _ve_jitted(factors: List[Factor]):
    # elimination order cannot be used as static argument
    return variable_elimination(factors, elimination_order)

result_factor, log_evidence = _ve_jitted(fs)

print(result_factor.table)
print(log_evidence)