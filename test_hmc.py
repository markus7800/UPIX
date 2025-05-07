
from dccxjax import *
import dccxjax.distributions as dist
import matplotlib.pyplot as plt
import jax.numpy as jnp
from time import time

@model
def normal():
    # sample("X", dist.Normal(0., 1.))
    sample("X", dist.Uniform(0., 1.))


m = normal()
slp = SLP_from_branchless_model(m)

n_chains = 10
n_samples_per_chain = 10_000

mcmc_config = MCMC(
    slp,
    MCMCStep(AllVariables(), HMC(10,0.1,unconstrained=True)),
    # MCMCStep(AllVariables(), RW(gaussian_random_walk(1.0))),
    n_chains,
    collect_inference_info=True,
    progress_bar=True,
    return_map=lambda x: x.position
)

init_trace, init_log_prob = broadcast_jaxtree(m.generate(jax.random.PRNGKey(0)), (mcmc_config.n_chains,))
init_trace = StackedTrace(init_trace, mcmc_config.n_chains)

result, all_positions = mcmc_config.run(jax.random.PRNGKey(0), init_trace, init_log_prob, n_samples_per_chain=n_samples_per_chain)
assert result.infos is not None
for info in result.infos:
    print(summarise_mcmc_info(info, n_samples_per_chain))
print(result)

all_positions = StackedTraces(all_positions, n_samples_per_chain, mcmc_config.n_chains)

plt.hist(all_positions.unstack().data["X"], density=True, bins=100)
xs = jnp.linspace(-5.,5., 100)
ps = jnp.exp(dist.Normal(0.,1.).log_prob(xs))
plt.plot(xs, ps)
plt.show()