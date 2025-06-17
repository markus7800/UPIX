import sys
sys.path.append(".")

from dccxjax import *
import dccxjax.distributions as dist
from typing import List
import jax
import jax.numpy as jnp

@model
def urn(obs: IntArray, biased: bool):
    N = sample("N", dist.Poisson(6.))
    balls: List[IntArray] = []
    for n in range(N):
        ball = sample(f"ball_{n}", dist.Bernoulli(0.5))
        balls.append(ball)
    balls_arr = jnp.hstack(tuple(balls))
    draws: List[IntArray] = []
    for k in range(len(obs)):
        draw = sample(f"draw_{k}", dist.DiscreteUniform(0,N-1))
        draws.append(draw)
    for k in range(len(obs)):
        if biased:
            sample(f"obs_{k}", dist.Bernoulli(jax.lax.select(balls_arr[draws[k]] == 1., 0.8, 0.2)), obs[k])
        else:
            logfactor(jax.lax.select(balls_arr[draws[k]] == obs[k], 0., -jnp.inf), f"factor_{k}")
            
# obs = jnp.array([0,1])
obs = jnp.array([0,1] * 5)
m = urn(obs, True)

def _get_n(slp: SLP) -> int:
    return int(slp.decision_representative["N"].item())
m.set_slp_formatter(lambda slp: f"N={_get_n(slp)}")
m.set_slp_sort_key(_get_n)

from pprint import pprint

from dccxjax.infer.exact import make_all_factors_fn, get_supports, compute_factors, variable_elimination
from dccxjax.infer.greedy_elimination_order import get_greedy_elimination_order
from dccxjax.core.samplecontext import GenerateCtx

with GenerateCtx(jax.random.PRNGKey(0), {"N": jnp.array(3,int)}) as ctx:
    m()
    X = ctx.X
    pprint(X)
    slp = slp_from_decision_representative(m, X)
    print(slp.formatted())
    # factors_fn = make_all_factors_fn(slp)
    # pprint(factors_fn(slp.decision_representative))
    # pprint(get_supports(slp))
    factors = compute_factors(slp)
    # pprint(factors)
    # elimination_order_set = set(slp.decision_representative.keys())
    # elimination_order_set.discard("N")
    # elimination_order = list(elimination_order_set)
    elimination_order = get_greedy_elimination_order(factors, ["N"])
    result, log_evidence = variable_elimination(factors, elimination_order)
    print(result, result.table, log_evidence)


    
exit()


for i in range(1):
    X, lp = m.generate(jax.random.PRNGKey(i))
exit()
class Config(MCDCC[T]):
    def run_inference(self, slp: SLP, rng_key: jax.Array) -> InferenceResult:
        raise NotImplementedError
    def estimate_log_weight(self, slp: SLP, rng_key: PRNGKey) -> LogWeightEstimate:
        raise NotImplementedError

config = Config(m, n_init_samples=100, verbose=2)

config.run(jax.random.PRNGKey(0))