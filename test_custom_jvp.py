
from dccxjax.all import *
from dccxjax.core.branching_tracer import trace_branching, retrace_branching, BranchingDecisions
import dccxjax.distributions as dist
import jax
import jax.numpy as jnp

def log1pexp(x):
  return jnp.log(1. + jnp.exp(x))

log1pexp(3.)

from jax import make_jaxpr, grad

# print("GT")
# print(make_jaxpr((log1pexp))(1.))

# print("BRANCHING")
# decisions = BranchingDecisions()
# print(make_jaxpr(trace_branching((log1pexp), decisions))(1.))
# exit()

from jax import custom_jvp

@custom_jvp
def log1pexp2(x):
  return jnp.log(1. + jnp.exp(x))
  
@log1pexp2.defjvp
def log1pexp_jvp2(primals, tangents):
  x, = primals
  x_dot, = tangents
  ans = log1pexp(x)
  ans_dot = (1 - 1/(1 + jnp.exp(x))) * x_dot
  return ans, ans_dot

def f(x):
  a = log1pexp2(x)
  return branching(a)

print("GT CUSTOM_JVP")
# print(make_jaxpr((f))(1.))
f(1)
f(1)

print("BRANCHING CUSTOM_JVP")
decisions = trace_branching(f, 1.)
print("retrace_branching=", retrace_branching(f, decisions)(1.))
print(make_jaxpr((retrace_branching(f, decisions)))(1.))
print(decisions.to_human_readable())


print("GT CUSTOM_JVP")
print(make_jaxpr(grad(log1pexp2))(1.))

print("BRANCHING CUSTOM_JVP")
print(make_jaxpr(retrace_branching(grad(log1pexp2), decisions))(1.))



# from jax import custom_vjp

# @custom_vjp
# def f(x, y):
#   return jnp.sin(x) * y

# def f_fwd(x, y):
#   # Returns primal output and residuals to be used in backward pass by f_bwd.
#   return f(x, y), (jnp.cos(x), jnp.sin(x), y)

# def f_bwd(res, g):
#   cos_x, sin_x, y = res # Gets residuals computed in f_fwd
#   return (cos_x * g * y, sin_x * g)

# f.defvjp(f_fwd, f_bwd)

