
from upix.all import *
import upix.distributions as dist
import jax.numpy as jnp

import logging
# setup_logging(logging.DEBUG)

@model
def mm():
    sample("m1", dist.Normal(-1.,1.))
    sample("m2", dist.MultivariateNormal(0.,jnp.eye(2)))

m: Model = mm()

X = sample_from_prior(m, jax.random.key(0))
print(X)

slp = slp_from_decision_representative(m, X)

for (n_chains, collect_states) in [(1,False), (1,True), (4,False), (4,True)]:
    print(f"{n_chains=}, {collect_states=}")
    result = mcmc(slp, MCMCStep(AllVariables(), RandomWalk(gaussian_random_walk(0.5))), 10, n_chains, jax.random.key(0), collect_states=collect_states)
    for addr, values in result.items():
        print(addr, values.shape)

print("\n\n")

for (n_chains, collect_states) in [(1,False), (1,True), (4,False), (4,True)]:
    print(f"{n_chains=}, {collect_states=}")
    config = DCC_Config(10, n_chains, collect_states, 10, 10**6)
    result = dcc(m, MCMCStep(AllVariables(), RandomWalk(gaussian_random_walk(0.5))), jax.random.key(0), config)
    print()

