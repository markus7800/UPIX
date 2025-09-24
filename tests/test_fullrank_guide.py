
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from typing import Callable
from time import time
from upix.all import *
import upix.distributions as dist
import upix.distributions.constraints as constraints
from upix.infer.optimizers import Adagrad


@model
def normal(obs):
    x = sample("x", dist.Normal(0, 1))
    y = sample("y", dist.Normal(x, 1))
    # z = sample("z", dist.Normal(y, 1))
    sample("obs", dist.Normal(y,1), observed=obs)

m: Model = normal(1.)
slp = SLP_from_branchless_model(m)

n_samples_per_chain = 10_000
n_chains = 10

mcmc_obj = MCMC(slp, MCMCStep(AllVariables(), HMC(10, 0.1)), n_chains, return_map=lambda x: x.position)
_, traces = mcmc_obj.run(
    jax.random.key(0),
    StackedTrace(broadcast_jaxtree(slp.decision_representative, (n_chains,)), n_chains),
    n_samples_per_chain=n_samples_per_chain
)
traces = StackedTraces(traces, n_samples_per_chain, n_chains)

traces = traces.unstack().get()


# plt.scatter(traces["x"], traces["y"], alpha=0.1, s=1)
# plt.show()

X = jnp.vstack((traces["x"], traces["y"]))
print(jnp.mean(X,axis=1))
print(jnp.cov(X))


# g: Guide = MeanfieldNormalGuide(slp, AllVariables())

g: Guide = FullRankNormalGuide(slp, AllVariables())
print(g.mu)
print(g.L)

# exit()


advi = ADVI(slp, g, Adagrad(1.), 100, progress_bar=True)
last_state, elbo = advi.run(jax.random.key(0), n_iter=1_000)
plt.figure()
plt.plot(elbo)
plt.show()


g = advi.get_updated_guide(last_state)
if isinstance(g, FullRankNormalGuide):
    print(g.mu)
    print(g.L)
posterior = g.sample(jax.random.key(0), (100_000,))
X = jnp.vstack((posterior["x"], posterior["y"]))
print(jnp.mean(X,axis=1))
print(jnp.cov(X))
plt.figure()
plt.scatter(posterior["x"], posterior["y"], alpha=0.1, s=1)
plt.show()
