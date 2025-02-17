from dccxjax import *
import jax
import jax.numpy as jnp
import dccxjax.distributions as dist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List
from time import time

import logging
setup_logging(logging.WARNING)

def geometric_rec(p, i):
    u =  sample(f"b_{i}", dist.Uniform(0.,1.))
    if u < p:
        return i
    else:
        return geometric_rec(p, i+1)

def geometric(p):
    return geometric_rec(p, 0)

m = model(geometric)(0.5)

def find_i_max(slp: SLP):
    i_max = 0
    for addr in slp.decision_representative.keys():
        if addr.startswith("b_"):
            i_max = max(i_max, int(addr[2:]))
    return i_max
def formatter(slp: SLP):
    i_max = find_i_max(slp)
    return f"#b={i_max}"
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_i_max)

rng_key = jax.random.PRNGKey(0)
active_slps: List[SLP] = []
for _ in tqdm(range(1_000)):
    rng_key, key = jax.random.split(rng_key)
    X = sample_from_prior(m, key)
    slp = slp_from_decision_representative(m, X)

    if all(slp.path_indicator(X) == 0 for slp in active_slps):
        active_slps.append(slp)

        # slp_to_mcmc_step[slp] = get_inference_regime_mcmc_step_for_slp(slp, deepcopy(regime), config.n_chains, config.collect_intermediate_chain_states)
active_slps = sorted(active_slps, key=m.slp_sort_key)
active_slps = active_slps[:10]

for slp in active_slps:
    print(slp.short_repr(), slp.formatted())