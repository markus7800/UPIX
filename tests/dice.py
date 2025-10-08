from upix.core import *
import upix.distributions as dist
import jax
import jax.numpy as jnp
from upix.infer.exact import *
from typing import List
from time import time

def diamond_rec(s1, N: int):
    route = sample(f"route_{N}", dist.Bernoulli(0.5))
    s2 = jax.lax.select(route, s1, 0)
    s3 = jax.lax.select(route, 0, s1)
    drop = sample(f"drop_{N}", dist.Bernoulli(0.001))
    # res = s2 | (s3 & (1-drop))
    res = sample(f"res_{N}", dist.Dirac(s2 | (s3 & (1-drop))))
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
    diamond_rec(jnp.array(1,int), N-1)
    
    
slp = SLP_from_branchless_model(diamond(100))
    
t0 = time()
supports = get_supports(slp)
print(f"Generated supports in {time()-t0:.3f}s")
# print(supports)



t0 = time()
fs = compute_factors(slp, supports, False)
print(f"Computed factors in {time()-t0:.3f}s")
# for f in fs:
#     print(f, f.table)

marginal_variables = ["res_0"]
elimination_order = get_greedy_elimination_order(fs, marginal_variables)

@jax.jit
def _ve_jitted(factors: List[Factor]):
    # elimination order cannot be used as static argument
    return variable_elimination(factors, elimination_order)

t0 = time()
result_factor, log_evidence = _ve_jitted(fs)
print(f"Performed VE in {time()-t0:.3f}s")

print(result_factor.table)
print(log_evidence)