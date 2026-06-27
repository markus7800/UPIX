
from upix.core import *
import upix.distributions as dist
from upix.distributions import Distribution
import numpyro.distributions as numpyro_dists
from typing import List, Dict, Optional
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

class Binomial(Distribution[IntArray,IntArrayLike]):
    def __init__(self, n: IntArrayLike, p: FloatArrayLike):
        super().__init__(numpyro_dists.Binomial(n, p)) # type: ignore

@model
def urn(obs: IntArray, biased: bool):
    N = sample("N", dist.Poisson(6.))
    n_black = sample("n_black", Binomial(N, 0.5))
    for k in range(len(obs)):
        black = sample(f"draw_{k}", dist.Bernoulli(n_black/N))
        if biased:
            sample(f"obs_{k}", dist.Bernoulli(jax.lax.select(black == 1., 0.8, 0.2)), obs[k])
        else:
            logfactor(jax.lax.select(black == obs[k], 0., -jnp.inf), f"factor_{k}")
            

def _get_n(slp: SLP) -> int:
    return int(slp.decision_representative["N"].item())


from upix.infer import ExactDCC, VariableSelector, SingleVariable, PrefixSelector, Factor
from upix.infer import compute_factors_iteratively, compute_factors_vmapped_single_pass, compute_factors_optimised
    
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
        return compute_factors_vmapped_single_pass(slp, supports, False)
        
    
    