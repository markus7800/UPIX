import sys
sys.path.insert(0, ".")

from dccxjax.all import *
from dccxjax.core import SLP
import dccxjax.distributions as dist
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

import logging

from dccxjax.infer.mcmc import MCMCRegime
setup_logging(logging.WARNING)

@model
def simple_branching_model(p):
    u = sample("u", dist.Uniform())
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

class DCCConfig(MCMCDCC[DCC_COLLECT_TYPE]):
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        return MCMCStep(AllVariables(), RW(gaussian_random_walk(0.5)))
    

dcc_obj = DCCConfig(m, verbose=2,
              mcmc_n_chains=10,
              mcmc_n_samples_per_chain=5_000,
              mcmc_collect_for_all_traces=True)

result = dcc_obj.run(jax.random.PRNGKey(0))
result.pprint()

plot_histogram(result, "u")
plot_trace(result, "u")
plot_histogram_by_slp(result, "u")
plt.show()
plot_histogram(result, "m1")
plot_trace(result, "m1")
plot_histogram_by_slp(result, "m1")
plt.show()
plot_histogram(result, "m2")
plot_trace(result, "m2")
plot_histogram_by_slp(result, "m2")
plt.show()

exit()

@model
def simple_branching_model_2(p):
    b = sample("b", dist.Bernoulli(p))
    if b:
        m = sample("m1", dist.Normal(-1.,1.))
    else:
        m = sample("m2", dist.Normal(2.,0.5))
    sample("y", dist.Normal(m, 0.5), observed=0.5)



p = 0.7

m2 = simple_branching_model_2(p)
m2.set_slp_formatter(HumanReadableDecisionsFormatter())


class DCCConfig2(MCMCDCC[DCC_COLLECT_TYPE]):
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        return MCMCSteps(
            # MCMCStep(SingleVariable("b"), RW(lambda b: dist.Bernoulli(0.1 * b + 0.9 * (1-b)))),
            MCMCStep(PrefixSelector("m"), RW(gaussian_random_walk(0.5)))
        )
    

dcc_obj2 = DCCConfig2(m2, verbose=2,
              mcmc_n_chains=10,
              mcmc_n_samples_per_chain=100_000,
              mcmc_collect_for_all_traces=False)

result = dcc_obj2.run(jax.random.PRNGKey(0))
result.pprint()