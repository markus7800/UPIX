import sys
sys.path.append(".")

from dccxjax import *
import dccxjax.distributions as dist
from typing import List
import jax
import jax.numpy as jnp

@model
def urn(K: int, obs: IntArray, biased: bool):
    N = sample("N", dist.Poisson(6.))
    balls: List[IntArray] = []
    for n in range(N):
        ball = sample(f"ball_{n}", dist.Bernoulli(0.5))
        balls.append(ball)
    balls_arr = jnp.hstack(tuple(balls))
    draws: List[IntArray] = []
    for k in range(K):
        draw = sample(f"draw_{k}", dist.DiscreteUniform(0,N-1))
        draws.append(draw)
    for k in range(K):
        if biased:
            sample(f"obs_{k}", dist.Bernoulli(jax.lax.select(balls_arr[draws[k]] == 1., 0.8, 0.2)), obs[k])
        else:
            logfactor(jax.lax.select(balls[draws[k]] == obs[k], 0., -jnp.inf))
            
K = 10
obs = jnp.array([0,1] * (K//2))
m = urn(K, obs, True)

def _get_n(slp: SLP) -> int:
    return int(slp.decision_representative["N"].item())
m.set_slp_formatter(lambda slp: f"N={_get_n(slp)}")
m.set_slp_sort_key(_get_n)


def f(x):
    if x > 0:
        return x*x
    else:
        return -x*x

from dccxjax.core.branching_tracer import trace_branching, retrace_branching

out, decisions = trace_branching(f, jnp.array(0.1,float))
print(out, decisions.to_human_readable())

# this does not work:
# out, decisions = trace_branching(jax.vmap(f), jnp.array([0.1],float))

out, _ = jax.vmap(retrace_branching(f, decisions))(jnp.array([0.1],float))
print(out)

# this does not work:
# out = retrace_branching(jax.vmap(f), decisions)(jnp.array([0.1],float))
# print(out)


from dccxjax.infer.exact import get_factors
for i in range(1):
    X, lp = m.generate(jax.random.PRNGKey(i))
    slp = slp_from_decision_representative(m, X)
    print(slp.formatted())
    get_factors(slp)


exit()
class Config(MCDCC[T]):
    def run_inference(self, slp: SLP, rng_key: jax.Array) -> InferenceResult:
        raise NotImplementedError
    def estimate_log_weight(self, slp: SLP, rng_key: PRNGKey) -> LogWeightEstimate:
        raise NotImplementedError

config = Config(m, n_init_samples=100, verbose=2)

config.run(jax.random.PRNGKey(0))