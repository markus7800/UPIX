from upix.all import *
import upix.distributions as dist
import jax.numpy as jnp
import matplotlib.pyplot as plt

import logging
setup_logging(logging.WARNING)

@model
def mm(p):
    u = sample("u", dist.Uniform())
    # print(f"{u=}")
    if u < p:
        m = sample("m1", dist.Normal(-1.,1.))
    else:
        m = sample("m2", dist.Normal(2.,0.5))
    sample("y", dist.Normal(m, 0.5), observed=0.5)

m: Model = mm(0.3)


p = 0.7

m: Model = mm(p) # type: ignore
m.set_slp_formatter(HumanReadableDecisionsFormatter())


U = jnp.linspace(0.,1.,1000)
M = jnp.linspace(-6.,6.,5000)
Us, Ms = jnp.meshgrid(U, M)

prior1 = jnp.exp(dist.Uniform().log_prob(Us) + dist.Normal(-1.,1.).log_prob(Ms))
prior2 = jnp.exp(dist.Uniform().log_prob(Us) + dist.Normal(2.,0.5).log_prob(Ms))
prior = prior2.at[Us < p].set(prior1[Us < p])

likeli = jnp.exp(dist.Normal(Ms, 0.5).log_prob(0.5))


active_slps: List[SLP] = []
for key in range(10):
    rng_key = jax.random.key(key)
    X = sample_from_prior(m, rng_key)
    slp = slp_from_decision_representative(m, X)

    if all(slp.path_indicator(X) == 0 for slp in active_slps):
        active_slps.append(slp)
#%%
print()
print()
print("active_slps: ")
for slp in active_slps:
    print(slp)
    result = coordinate_ascent(slp, 1., 100, 2, jax.random.key(0))
    print(result)
#     plt.plot(result["u"])
# plt.show()


plt.pcolormesh(Ms, Us, prior * likeli)
plt.show()