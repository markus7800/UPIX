
from dccxjax.all import *
import dccxjax.distributions as dist
import matplotlib.pyplot as plt
import jax.numpy as jnp
from time import time

@model
def normal():
    # sample("X", dist.Normal(0., 1.))
    sample("X", dist.Uniform(0., 1.))
    sample("Y", dist.Normal(0., 1.))


m = normal()
slp = SLP_from_branchless_model(m)

n_chains = 10
n_samples_per_chain = 10_000

mcmc_config = MCMC(
    slp,
    # MCMCStep(AllVariables(), HMC(10,0.1,unconstrained=False)),
    # MCMCStep(AllVariables(), DHMC(10,0.05,0.15,SingleVariable("X"),unconstrained=False)),
    MCMCStep(AllVariables(), DHMC(10,0.05,0.15,unconstrained=False)),
    # MCMCStep(AllVariables(), RW(gaussian_random_walk(1.0))),
    n_chains,
    collect_inference_info=True,
    progress_bar=True,
    return_map=lambda x: x.position
)

init_trace, init_log_prob = broadcast_jaxtree(m.generate(jax.random.key(0)), (mcmc_config.n_chains,))
init_trace = StackedTrace(init_trace, mcmc_config.n_chains)

result, all_positions = mcmc_config.run(jax.random.key(0), init_trace, init_log_prob, n_samples_per_chain=n_samples_per_chain)
assert result.infos is not None
for info in result.infos:
    print(summarise_mcmc_info(info, n_samples_per_chain))

all_positions = StackedTraces(all_positions, n_samples_per_chain, mcmc_config.n_chains)

x_sampled = all_positions.unstack().get()["X"]
plt.hist(x_sampled, density=True, bins=100)
xs = jnp.linspace(x_sampled.min()-0.1,x_sampled.max()+0.1, 1000)
# ps = jnp.exp(dist.Normal(0.,1.).log_prob(xs))
ps = jnp.exp(jax.vmap(slp.log_prob)({"X": xs, "Y": jnp.full_like(xs, 0.)}))
ps = ps / jnp.trapezoid(ps, xs)
plt.plot(xs, ps)
# plt.show()

y_sample = all_positions.unstack().get()["Y"]
plt.hist(y_sample, density=True, bins=100)
ys = jnp.linspace(y_sample.min()-0.1,y_sample.max()+0.1, 1000)
ps = jnp.exp(jax.vmap(slp.log_prob)({"X": jnp.full_like(ys, 0.), "Y": ys}))
ps = ps / jnp.trapezoid(ps, ys)
plt.plot(ys, ps)
plt.show()