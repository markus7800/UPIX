#%%
from dccxjax.all import *
import dccxjax.distributions as dist
import matplotlib.pyplot as plt
import jax.numpy as jnp
from time import time

import logging
setup_logging(logging.DEBUG)

@model
def normal(observed):
    X = sample("X", dist.Normal(0.,1.))
    Y = sample("Y", dist.Normal(X, 1.), observed=observed)


m: Model = normal(None)
slp = convert_model_to_SLP(m)

#%%
t0 = time()
result = mcmc(
    slp,
    MCMCStep(AllVariables(), RW(gaussian_random_walk(0.5))),
    10**5,
    1,
    jax.random.PRNGKey(0)
    )
t1 = time()

print(result)
print(f"in {t1-t0:.3f} seconds")
exit()
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
