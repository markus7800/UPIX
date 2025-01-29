#%%
from dccxjax import *
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import jax.numpy as jnp

import logging
setup_logging(logging.DEBUG)

@model
def normal(observed):
    X = sample("X", dist.Normal(0.,1.))
    Y = sample("Y", dist.Normal(X, 1.), observed=observed)


m: Model = normal(None)
slp = convert_model_to_SLP(m)

#%%
result = mcmc(
    slp,
    InferenceStep(AllVariables(), RW(gaussian_random_walk(0.5))),
    100_000,
    1,
    jax.random.PRNGKey(0)
    )

print(result)
#%%
plt.hist(result["X"], density=True, bins=50)
xs = jnp.linspace(-5.,5.,500)
ps = jnp.exp(dist.Normal(0.,1.).log_prob(xs))
plt.plot(xs, ps)
plt.show()
exit()
#%%
plt.hist(result["Y"], density=True, bins=50)
ys = jnp.linspace(-5.,5.,500)
ps = jnp.exp(dist.Normal(0.,jnp.sqrt(2.)).log_prob(xs))
plt.plot(ys, ps)
plt.show()
# %%
