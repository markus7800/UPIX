import sys
sys.path.append(".")

from dccxjax import *
import dccxjax.distributions as dist
from typing import List
import jax
import jax.numpy as jnp
import time

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
            
obs = jnp.array([0,1] * 5)
m = urn(obs, True)

def _get_n(slp: SLP) -> int:
    return int(slp.decision_representative["N"].item())
m.set_slp_formatter(lambda slp: f"N={_get_n(slp)}")
m.set_slp_sort_key(_get_n)

from pprint import pprint

from dccxjax.infer.exact import make_all_factors_fn, get_supports, compute_factors, variable_elimination, Factor
from dccxjax.infer.greedy_elimination_order import get_greedy_elimination_order
from dccxjax.core.samplecontext import GenerateCtx
from dccxjax.utils import to_shaped_arrays_str_short

def compute_factors_fast(slp: SLP, jit: bool = True):
    all_factors_fn = make_all_factors_fn(slp)
    factor_prototypes = all_factors_fn(slp.decision_representative)
    supports = get_supports(slp)
    def _get_support(addr: str) -> IntArray:
        s = supports[addr]
        if s is None:
            return jnp.array([slp.decision_representative[addr]])
        else:
            return s
        
    # order: N, ball_{}, draw_{}
    N = int(slp.decision_representative["N"].item())
    K = len(slp.model.args[0])
    Bs = [f"ball_{n}" for n in range(N)]
    Bs_support = list(map(_get_support, Bs))
    N_support = _get_support("N")
    D_support = _get_support("draw_0")
    addresses_prototype = ["N"] + Bs + ["draw_"]
    factor_variable_supports = [N_support] + Bs_support + [D_support]
    
    meshgrids = jnp.meshgrid(*factor_variable_supports, indexing="ij")
    meshgrid_shape = meshgrids[0].shape
        
    X = {}
    X["N"] = meshgrids[0].reshape(-1)
    for n in range(N):
        X[f"ball_{n}"] = meshgrids[n+1].reshape(-1)
    for k in range(K):
        X[f"draw_{k}"] = meshgrids[-1].reshape(-1)
    
    
    @jax.vmap
    def _factor_fn(_partial_X: Trace) -> List[FloatArray]:
        return [val for val, _  in all_factors_fn(_partial_X)]
    factor_fn = jax.jit(_factor_fn) if jit else _factor_fn

    res = factor_fn(X)
    
    factors: List[Factor] = []
    for i, (_, addresses) in enumerate(factor_prototypes):
        selector = []
        for addr in addresses_prototype:
            if addr == "draw_" and any(factor_address.startswith(addr) for factor_address in addresses):
                selector.append(slice(None))
            elif addr in addresses:
                selector.append(slice(None))
            else:
                selector.append(0)
        factor_table = res[i].reshape(meshgrid_shape)[*selector]
        factor = Factor(addresses, factor_table)
        # print(factor)
        factors.append(factor)

    return factors

    
    
    
lp = []
for N in range(1,15+1):
    with GenerateCtx(jax.random.PRNGKey(0), {"N": jnp.array(N,int)}) as ctx:
        m()
        X = ctx.X
        # pprint(X)
        slp = slp_from_decision_representative(m, X)
        print(slp.formatted())
        # factors_fn = make_all_factors_fn(slp)
        # pprint(factors_fn(slp.decision_representative))
        # pprint(get_supports(slp))
        t0 = time.time()
        # factors = compute_factors(slp, True)
        factors = compute_factors_fast(slp, True)
        t1 = time.time()
        print(f"Computed factors in {t1-t0:.3f}s")
        # pprint(factors)
        # elimination_order_set = set(slp.decision_representative.keys())
        # elimination_order_set.discard("N")
        # elimination_order = list(elimination_order_set)
        elimination_order = get_greedy_elimination_order(factors, ["N"])
        t2 = time.time()
        print(f"Computed elimination_order in {t2-t1:.3f}s")
        @jax.jit
        def _ve(factors: List[Factor]):
            return variable_elimination(factors, elimination_order)
        result, log_evidence = _ve(factors)
        t3 = time.time()
        print(f"Computed variable_elimination in {t3-t2:.3f}s")
        print(result, result.table, log_evidence)
        lp.append(log_evidence)
       
lp = jnp.hstack(lp)
print(jax.scipy.special.logsumexp(lp))
lp = lp - jax.scipy.special.logsumexp(lp)
print(jnp.exp(lp))

    
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