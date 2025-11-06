from upix.core import *
import upix.distributions as dist
import jax
import jax.numpy as jnp
from upix.infer.exact import *
from typing import List
from time import time
from upix.infer import PrefixSelector

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
    
    
slp = SLP_from_branchless_model(diamond(10))
    
t0 = time()
supports = get_supports(slp)
print(f"Generated supports in {time()-t0:.3f}s")
# print(supports)



# t0 = time()
# fs = compute_factors(slp, supports, False)
# print(f"Computed factors in {time()-t0:.3f}s")

t0 = time()
fs = compute_factors_optimised(slp, [[PrefixSelector("drop"), PrefixSelector("res"), PrefixSelector("res"), PrefixSelector("route")]], supports, False)
print(f"Computed factors in {time()-t0:.3f}s")

for f in fs:
    print(f, f.table)

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


# Generated supports in 0.648s
# Computed factors in 2.898s
# Factor(['drop_0'], (2,)) [-1.0005004e-03 -6.9077554e+00]
# Factor(['drop_1'], (2,)) [-1.0005004e-03 -6.9077554e+00]
# Factor(['drop_2'], (2,)) [-1.0005004e-03 -6.9077554e+00]
# Factor(['drop_3'], (2,)) [-1.0005004e-03 -6.9077554e+00]
# Factor(['drop_4'], (2,)) [-1.0005004e-03 -6.9077554e+00]
# Factor(['drop_5'], (2,)) [-1.0005004e-03 -6.9077554e+00]
# Factor(['drop_6'], (2,)) [-1.0005004e-03 -6.9077554e+00]
# Factor(['drop_7'], (2,)) [-1.0005004e-03 -6.9077554e+00]
# Factor(['drop_8'], (2,)) [-1.0005004e-03 -6.9077554e+00]
# Factor(['drop_9'], (2,)) [-1.0005004e-03 -6.9077554e+00]
# Factor(['drop_0', 'res_0', 'res_1', 'route_0'], (2, 2, 2, 2)) [[[[  0.   0.]
#    [-inf -inf]]

#   [[-inf -inf]
#    [  0.   0.]]]


#  [[[  0.   0.]
#    [  0. -inf]]

#   [[-inf -inf]
#    [-inf   0.]]]]
# Factor(['drop_1', 'res_1', 'res_2', 'route_1'], (2, 2, 2, 2)) [[[[  0.   0.]
#    [-inf -inf]]

#   [[-inf -inf]
#    [  0.   0.]]]


#  [[[  0.   0.]
#    [  0. -inf]]

#   [[-inf -inf]
#    [-inf   0.]]]]
# Factor(['drop_2', 'res_2', 'res_3', 'route_2'], (2, 2, 2, 2)) [[[[  0.   0.]
#    [-inf -inf]]

#   [[-inf -inf]
#    [  0.   0.]]]


#  [[[  0.   0.]
#    [  0. -inf]]

#   [[-inf -inf]
#    [-inf   0.]]]]
# Factor(['drop_3', 'res_3', 'res_4', 'route_3'], (2, 2, 2, 2)) [[[[  0.   0.]
#    [-inf -inf]]

#   [[-inf -inf]
#    [  0.   0.]]]


#  [[[  0.   0.]
#    [  0. -inf]]

#   [[-inf -inf]
#    [-inf   0.]]]]
# Factor(['drop_4', 'res_4', 'res_5', 'route_4'], (2, 2, 2, 2)) [[[[  0.   0.]
#    [-inf -inf]]

#   [[-inf -inf]
#    [  0.   0.]]]


#  [[[  0.   0.]
#    [  0. -inf]]

#   [[-inf -inf]
#    [-inf   0.]]]]
# Factor(['drop_5', 'res_5', 'res_6', 'route_5'], (2, 2, 2, 2)) [[[[  0.   0.]
#    [-inf -inf]]

#   [[-inf -inf]
#    [  0.   0.]]]


#  [[[  0.   0.]
#    [  0. -inf]]

#   [[-inf -inf]
#    [-inf   0.]]]]
# Factor(['drop_6', 'res_6', 'res_7', 'route_6'], (2, 2, 2, 2)) [[[[  0.   0.]
#    [-inf -inf]]

#   [[-inf -inf]
#    [  0.   0.]]]


#  [[[  0.   0.]
#    [  0. -inf]]

#   [[-inf -inf]
#    [-inf   0.]]]]
# Factor(['drop_7', 'res_7', 'res_8', 'route_7'], (2, 2, 2, 2)) [[[[  0.   0.]
#    [-inf -inf]]

#   [[-inf -inf]
#    [  0.   0.]]]


#  [[[  0.   0.]
#    [  0. -inf]]

#   [[-inf -inf]
#    [-inf   0.]]]]
# Factor(['drop_8', 'res_8', 'res_9', 'route_8'], (2, 2, 2, 2)) [[[[  0.   0.]
#    [-inf -inf]]

#   [[-inf -inf]
#    [  0.   0.]]]


#  [[[  0.   0.]
#    [  0. -inf]]

#   [[-inf -inf]
#    [-inf   0.]]]]
# Factor(['drop_9', 'res_9', 'route_9'], (2, 2, 2)) [[[-inf -inf]
#   [  0.   0.]]

#  [[  0. -inf]
#   [-inf   0.]]]
# Factor(['route_0'], (2,)) [-0.6931472 -0.6931472]
# Factor(['route_1'], (2,)) [-0.6931472 -0.6931472]
# Factor(['route_2'], (2,)) [-0.6931472 -0.6931472]
# Factor(['route_3'], (2,)) [-0.6931472 -0.6931472]
# Factor(['route_4'], (2,)) [-0.6931472 -0.6931472]
# Factor(['route_5'], (2,)) [-0.6931472 -0.6931472]
# Factor(['route_6'], (2,)) [-0.6931472 -0.6931472]
# Factor(['route_7'], (2,)) [-0.6931472 -0.6931472]
# Factor(['route_8'], (2,)) [-0.6931472 -0.6931472]
# Factor(['route_9'], (2,)) [-0.6931472 -0.6931472]
# Performed VE in 0.178s
# [-5.300567e+00 -5.001051e-03]
# 2.454035e-07