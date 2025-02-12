import sys
sys.path.insert(0, ".")

from dccxjax import *
import jax
import numpyro.distributions as dist

import logging
setup_logging(logging.WARNING)

@model
def pedestrian():
    start = sample("start", dist.Uniform(0.,3.))
    position = start
    distance = 0.
    t = 0
    while (position > 0) & (distance < 10):
        t += 1
        step = sample(f"step_{t}", dist.Uniform(-1.,1.))
        position += step
        distance += jax.lax.abs(step)
    sample("obs", dist.Normal(distance, 0.1), observed=1.1)
    return start


m: Model = pedestrian()

config = DCC_Config(
    n_samples_from_prior = 10,
    n_chains = 1024,
    collect_intermediate_chain_states = True,
    n_samples_per_chain = 128,
    n_samples_for_Z_est = 10**6
)
result = dcc(m, InferenceStep(AllVariables(), RandomWalk(gaussian_random_walk(0.1))), jax.random.PRNGKey(0), config)

plot_histogram(result, "start")
plt.show()