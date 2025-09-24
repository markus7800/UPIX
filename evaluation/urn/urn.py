
from upix.core import *
import upix.distributions as dist
from typing import List, Dict, Optional
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm


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
            

def _get_n(slp: SLP) -> int:
    return int(slp.decision_representative["N"].item())


from upix.infer import ExactDCC, VariableSelector, SingleVariable, PrefixSelector, Factor, compute_factors_optimised
    
class Config(ExactDCC):
    
    def __init__(self, model: Model, *ignore, verbose=0, **config_kwargs) -> None:
        super().__init__(model, *ignore, verbose=verbose, **config_kwargs)
        self.N_max = self.config["N_max"]
        self.share_progress_bar = False
    
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        for N in range(1,self.N_max+1):
            X, _ = self.model.generate(jax.random.key(0), {"N": jnp.array(N,int)})
            slp = slp_from_decision_representative(self.model, X)
            tqdm.write(f"Make SLP {slp.formatted()} active.")
            active_slps.append(slp)
            
    def get_query_variables(self, slp: SLP) -> List[str]:
        return []
    
    def get_factors(self, slp: SLP, supports: Dict[str, Optional[IntArray]]) -> List[Factor]:
        N = int(slp.decision_representative["N"].item())
        selectors: List[VariableSelector] = []
        selectors.append(SingleVariable("N"))
        for addr in sorted([f"ball_{n}" for n in range(N)]):
            selectors.append(SingleVariable(addr))
        selectors.append(PrefixSelector("draw_"))
        return compute_factors_optimised(slp, [selectors], supports, True)
        
    
    