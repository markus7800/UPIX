import sys
sys.path.append("evaluation")
from parse_args import parse_args_and_setup
args = parse_args_and_setup()
from setup_parallelisation import get_parallelisation_config
        
from dccxjax.all import *
import dccxjax.distributions as dist
from typing import List, Dict, Optional
import jax
import jax.numpy as jnp
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


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

    
    
class Config(ExactDCC):
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        for N in range(1,20+1):
            X, _ = m.generate(jax.random.key(0), {"N": jnp.array(N,int)})
            slp = slp_from_decision_representative(m, X)
            tqdm.write(f"Make SLP {slp.formatted()} active.")
            active_slps.append(slp)
    def get_query_variables(self, slp: SLP) -> List[str]:
        return []
    
    # def get_factors(self, slp: SLP) -> List[Factor]:
    #     t0 = time.time()
    
    #     N = int(slp.decision_representative["N"].item())
        
    #     selectors_1 = [SingleVariable("N"), PrefixSelector("draw_")]
        
    #     selectors_2: List[VariableSelector] = []
    #     for addr in sorted([f"ball_{n}" for n in range(N)]):
    #         selectors_2.append(SingleVariable(addr))
    #     selectors_2.append(PrefixSelector("draw_"))
        
    #     factors = compute_factors_optimised(slp, [selectors_1, selectors_2], True)

    #     t1 = time.time()
        
    #     if self.verbose >= 2:
    #         tqdm.write(f"Computed factors in {t1-t0:.3f}s")
    #     return factors
    
    def get_factors(self, slp: SLP, supports: Dict[str, Optional[IntArray]]) -> List[Factor]:        
        N = int(slp.decision_representative["N"].item())
        selectors: List[VariableSelector] = []
        selectors.append(SingleVariable("N"))
        for addr in sorted([f"ball_{n}" for n in range(N)]):
            selectors.append(SingleVariable(addr))
        selectors.append(PrefixSelector("draw_"))
        return compute_factors_optimised(slp, [selectors], supports, True)
        
    
    
config = Config(m, verbose=2,
    parallelisation = get_parallelisation_config(args),
    jit_inference=True,
    share_progress_bar=False
)

result = timed(config.run)(jax.random.key(0))
result.pprint(sortkey="slp")

gt = jnp.load("evaluation/urn/gt_ps.npy")
log_Zs = jnp.array([log_Z for (slp, log_Z) in result.get_log_weights_sorted(sortkey="slp")])


assert (jnp.array([slp.decision_representative["N"] for (slp, log_Z) in result.get_log_weights_sorted(sortkey="slp")]) == jnp.arange(1,len(log_Zs)+1)).all()


ps = jnp.exp(log_Zs - jax.scipy.special.logsumexp(log_Zs))
# print(ps)
# print("vs")
# print(gt[:len(ps)])

err = jnp.abs(ps - gt[:len(ps)])
plt.plot(err)
plt.show()

print("Max err: ", jnp.max(err))
