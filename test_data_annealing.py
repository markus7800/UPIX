import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Callable
from time import time
from dccxjax.infer.smc import *

from dccxjax import *
from dccxjax.infer.ais import *
from dccxjax.infer.mcmc import get_mcmc_kernel, MCMCState
import dccxjax.distributions as dist

import logging
setup_logging(logging.DEBUG)

obs = jax.random.normal(jax.random.PRNGKey(0), (100,)) + 2

@model
def normal(obs):
    x = sample("x", dist.Normal(0, 1))
    sample("y", dist.Normal(x, 1.0), observed=obs)

m: Model = normal(obs)
slp = SLP_from_branchless_model(m)


x_range = jnp.linspace(-4,4,1000)
log_joint = jax.vmap(slp.log_prob)({"x": x_range})
log_Z = jax.scipy.special.logsumexp(log_joint) + jnp.log(x_range[1] - x_range[0])
log_posterior = log_joint - log_Z
posterior = jnp.exp(log_posterior)
print(log_Z, jnp.trapezoid(posterior, x_range))
# plt.plot(x_range, posterior)
# plt.show()

temering_schedule = tempering_schedule_from_sigmoid(jnp.linspace(-25,25,10))
data_annealing_schedule = None

# temering_schedule = None
# data_annealing_schedule = data_annealing_schedule_from_range({"y": range(0,len(obs),1)})


n_particles = 1_000_000
smc_obj = SMC(
    slp,
    n_particles,
    temering_schedule,
    data_annealing_schedule,
    ReweightingType.BootstrapStaticPrior,
    MultinomialResampling(ResampleType.Adaptive, ResampleTime.BeforeMove),
    MCMCStep(SingleVariable("x"), RW(gaussian_random_walk(1.))),
    collect_inference_info=True,
    progress_bar=True
)
particles = {"x": jax.random.normal(jax.random.PRNGKey(0), (n_particles,))}
last_state, ess = smc_obj.run(jax.random.PRNGKey(0), StackedTrace(particles, n_particles))
last_state.log_particle_weights.block_until_ready()
print(get_log_Z_ESS(last_state.log_particle_weights))

# plt.hist(last_state.particles["x"], weights=jnp.exp(normalise_log_weights(last_state.log_particle_weights)), density=True, bins=100)
# plt.plot(x_range, posterior)
# plt.show()
# plt.plot(ess)
# plt.show()

# data_annealing_schedule = data_annealing_schedule_from_range({"y": range(0,10,3)})
# print(data_annealing_schedule)
# print(data_annealing_schedule.data_annealing["y"])