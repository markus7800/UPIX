import sys
sys.path.append(".")

from dccxjax import *
import dccxjax.distributions as dist
import jax

from trees import *
from analytical import *

import matplotlib.pyplot as plt

@model
def pyhlo(tree: Branch, rho: jax.Array):
    lam = sample("lambda", dist.Gamma(1.0, 1.0))
    mu = sample("mu", dist.Gamma(1.0, 0.5))
    
    # lam = sample("lambda", dist.Gamma(1.0, 1.0))
    # eps = sample("epsilon", dist.Uniform(0.,1.))
    # mu = lam * eps
    
    logfactor(exact_CRBD_loglikelihood(tree, lam, mu, rho))

bisse_32 = read_phyjson("evaluation/phylo/data/bisse_32.phyjson")
m = pyhlo(bisse_32, jnp.array(1.,float))
slp = SLP_from_branchless_model(m)

n_chains = 10
n_samples_per_chain = 10_000

mcmc_obj = MCMC(
    slp,
    # MCMCStep(AllVariables(), HMC(10, 0.1, unconstrained=True)),
    MCMCStep(AllVariables(), DHMC(10, 0.005, 0.015)),
    n_chains,
    collect_inference_info=True,
    return_map=lambda x: x.position,
    progress_bar=True
)

init_positions = StackedTrace(broadcast_jaxtree(slp.decision_representative, (n_chains,)), n_chains)

last_state, result = mcmc_obj.run(jax.random.key(0), init_positions, n_samples_per_chain=n_samples_per_chain)

assert last_state.infos is not None
print(summarise_mcmc_infos(last_state.infos, n_samples_per_chain))

result = StackedTraces(result, n_samples_per_chain, n_chains).unstack()

# plt.hist(result.get()["lambda"], bins=100, density=True, alpha=0.5)
# plt.hist(result.get()["mu"], bins=100, density=True, alpha=0.5)
# plt.show()

# plt.hist(result.get()["lambda"], bins=100, density=True, alpha=0.5)
# plt.hist(result.get()["epsilon"], bins=100, density=True, alpha=0.5)
# plt.hist(result.get()["epsilon"] * result.get()["lambda"], bins=100, density=True, alpha=0.5)
# plt.show()

n_lam = 1000
n_mu = 1000

lam = jnp.linspace(1e-5, 5.,n_lam)
mu = jnp.linspace(1e-5,10.,n_mu)
delta = (lam[1] - lam[0]) * (mu[1] - mu[0])

lam_mesh, mu_mesh = jnp.meshgrid(lam, mu)
lam_mesh = lam_mesh.reshape(-1)
mu_mesh = mu_mesh.reshape(-1)

logdensity = jax.vmap(slp.log_prior)({"lambda": lam_mesh, "mu": mu_mesh})
logdensity = logdensity.reshape((n_lam, n_mu))
print(jnp.exp(jax.scipy.special.logsumexp(logdensity)) * delta)

# plt.figure()
# plt.title("lam")
# plt.plot(lam, jnp.exp(jax.scipy.special.logsumexp(logdensity, axis=0)) * (mu[1] - mu[0]))
# plt.plot(lam, jnp.exp(dist.Gamma(1.0, 1.0).log_prob(lam)))
# plt.figure()
# plt.title("mu")
# plt.plot(mu, jnp.exp(jax.scipy.special.logsumexp(logdensity, axis=1)) * (lam[1] - lam[0]))
# plt.plot(mu, jnp.exp(dist.Gamma(1.0, 0.5).log_prob(mu)))
# plt.show()

lam = jnp.linspace(1e-5, 1., n_lam)
mu = jnp.linspace(1e-5, 1., n_mu)
delta = (lam[1] - lam[0]) * (mu[1] - mu[0])

lam_mesh, mu_mesh = jnp.meshgrid(lam, mu)
lam_mesh = lam_mesh.reshape(-1)
mu_mesh = mu_mesh.reshape(-1)

logdensity = jax.vmap(slp.log_prob)({"lambda": lam_mesh, "mu": mu_mesh})
logdensity = logdensity.reshape((n_lam, n_mu))
log_Z = jax.scipy.special.logsumexp(logdensity) + jnp.log(delta)
logdensity = logdensity - log_Z
print(log_Z)

plt.figure()

plt.plot(lam, jnp.exp(jax.scipy.special.logsumexp(logdensity, axis=0) + jnp.log(mu[1] - mu[0])), label="lam", color="tab:blue")
plt.plot(mu, jnp.exp(jax.scipy.special.logsumexp(logdensity, axis=1) + jnp.log(lam[1] - lam[0])), label="mu", color="tab:orange")

lam_sample = result.get()["lambda"]
mu_sample = result.get()["mu"]
plt.hist(lam_sample[lam_sample < lam.max()], bins=100, density=True, alpha=0.5, color="tab:blue")
plt.hist(mu_sample[mu_sample < mu.max()], bins=100, density=True, alpha=0.5, color="tab:orange")
plt.legend()
plt.show()