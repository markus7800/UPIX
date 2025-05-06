import sys
sys.path.insert(0, ".")

from dccxjax import *
from dccxjax.core import SLP
import dccxjax.distributions as dist
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dccxjax.infer.dcc2 import *


import logging

from dccxjax.infer.mcmc import InferenceRegime
setup_logging(logging.WARNING)

@model
def simple_branching_model(p):
    u = sample("u", dist.Uniform())
    # print(f"{u=}")
    if u < p:
        m = sample("m1", dist.Normal(-1.,1.))
    else:
        m = sample("m2", dist.Normal(2.,0.5))
    sample("y", dist.Normal(m, 0.5), observed=0.5)


p = 0.7

m = simple_branching_model(p)
m.set_slp_formatter(HumanReadableDecisionsFormatter())


# U = jnp.linspace(0.,1.,1000)
# M = jnp.linspace(-6.,6.,5000)
# Us, Ms = jnp.meshgrid(U, M)

# prior1 = jnp.exp(dist.Uniform().log_prob(Us) + dist.Normal(-1.,1.).log_prob(Ms))
# prior2 = jnp.exp(dist.Uniform().log_prob(Us) + dist.Normal(2.,0.5).log_prob(Ms))
# prior = prior2.at[Us < p].set(prior1[Us < p])

# likeli = jnp.exp(dist.Normal(Ms, 0.5).log_prob(0.5))

# Z1 = jnp.trapezoid(jnp.trapezoid(prior * likeli.at[Us >= p].set(0.), M, axis=0), U, axis=0)
# Z2 = jnp.trapezoid(jnp.trapezoid(prior * likeli.at[Us < p].set(0.), M, axis=0), U, axis=0)
# Z = jnp.trapezoid(jnp.trapezoid(prior * likeli, M, axis=0), U, axis=0)
# print(f"{Z=}")
# print(f"{Z1=}")
# print(f"{Z2=}")

class DCC(MCMCDCC[DCC_COLLECT_TYPE]):
    def get_MCMC_inference_regime(self, slp: SLP) -> InferenceRegime:
        return InferenceStep(AllVariables(), RW(gaussian_random_walk(0.5)))
    

dcc_obj = DCC(m, verbose=2,
              mcmc_n_chains=10,
              mcmc_n_samples_per_chain=100_000,
              mcmc_collect_for_all_traces=False)

result = dcc_obj.run(jax.random.PRNGKey(0))
print(result)