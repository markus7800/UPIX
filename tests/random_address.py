from upix.core import *
import upix.distributions as dist
import jax
import jax.numpy as jnp
from typing import List

@model
def diamond_rec():
    U = sample("U", dist.Uniform(0.,1.))
    # Z = sample(str(U), dist.Normal(0.,1.))
    Z = sample(str(U.item()), dist.Normal(0.,1.))
    # Z = sample(str(float(U)), dist.Normal(0.,1.))
    # Z = sample(str(branching(U)), dist.Normal(0.,1.))


m = diamond_rec()
m.set_slp_formatter(lambda slp: f"[U={slp.decision_representative["U"]}]")

discovered_slps: List[SLP] = []

# default samples from prior
rng_key = jax.random.key(0)
for _ in range(100):
    rng_key, key = jax.random.split(rng_key)
    trace = sample_from_prior(m, key)

    if all(slp.path_indicator(trace) == 0 for slp in discovered_slps):
        slp = slp_from_decision_representative(m, trace)
        print(f"Discovered SLP {slp.formatted(), slp.decision_representative}.")
        discovered_slps.append(slp)
