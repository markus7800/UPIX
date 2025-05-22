import sys
sys.path.insert(0, ".")

from data import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from dccxjax import *
import dccxjax.distributions as dist
import numpyro.distributions as numpyro_dist
from kernels import *
from dataclasses import fields
from tqdm.auto import tqdm
from dccxjax.core.branching_tracer import retrace_branching
from time import time
from data import get_data_autogp

import logging
setup_logging(logging.WARN)

# NOISE_Z = -3
NOISE_Z = None

@model
def prim_kernel(xs, ys):
    lengthscale_z = sample("lengthscale", dist.Normal(0.,1.))
    period_z = sample("period", dist.Normal(0.,1.))
    amplitude_z = sample("amplitude", dist.Normal(0.,1.))
    lengthscale = transform_param("lengthscale", lengthscale_z)
    period = transform_param("period", period_z)
    amplitude = transform_param("amplitude", amplitude_z)
    k = Periodic(lengthscale, period, amplitude)
    if NOISE_Z:
        noise_z = NOISE_Z
    else:
        noise_z = sample("noise", dist.Normal(0.,1.))
    noise = transform_param("noise", noise_z)
    cov_matrix = k.eval_cov_vec(xs) + (noise + 1e-5) * jnp.eye(xs.size)
    sample("obs", dist.MultivariateNormal(covariance_matrix=cov_matrix), ys)

def get_kernel(X: Trace):
    lengthscale = transform_param("lengthscale", X["lengthscale"])
    period = transform_param("period", X["period"])
    amplitude = transform_param("amplitude", X["amplitude"])
    noise = transform_param("noise", X["noise"] if "noise" in X else NOISE_Z) 
    k = Periodic(lengthscale, period, amplitude)
    return k, noise

xs = jnp.linspace(0.,1.,10)

# X, _ = prim_kernel(xs, None).generate(jax.random.PRNGKey(0), {"lengthscale": jnp.array(0.,float), "amplitude": jnp.array(0.,float), "period": jnp.array(0.,float)})
# print(X)
# lengthscale_gt = transform_param("lengthscale", X["lengthscale"])
# period_gt = transform_param("period", X["period"])
# amplitude_gt = transform_param("amplitude", X["amplitude"])
# # noise_gt = transform_param("noise", X["noise"])
# ys = X["obs"]

# k = Periodic(lengthscale_gt, period_gt, amplitude_gt)
# xs_pred = jnp.linspace(0.,2.,500)
# post = k.posterior_predictive(xs, ys, 1e-5, xs_pred, 1e-5)

# plt.scatter(xs, ys)
# plt.plot(xs_pred, post.numpyro_base.mean)
# plt.show()

# xs = jnp.linspace(0.,1.,100)
# ys = jnp.sin(xs*6*5) + jax.random.normal(jax.random.PRNGKey(0), (100,)) * 0.1


xs, xs_val, ys, ys_val = get_data_autogp()
xs = jax.random.permutation(jax.random.PRNGKey(0), xs)
ys = jax.random.permutation(jax.random.PRNGKey(0), ys)


slp = SLP_from_branchless_model(prim_kernel(xs, ys))
# plt.scatter(xs, ys)
# plt.show()

if False:
    n_chains = 10
    n_samples_per_chain = 1_000
    mcmc_obj = MCMC(
        slp,
        # MCMCStep(AllVariables(), RW(gaussian_random_walk(0.05),elementwise=True)),
        # MCMCStep(AllVariables(), RW(lambda _: dist.Normal(0.,1.), elementwise=True)),
        # MCMCSteps(MCMCStep(AllVariables(), RW(lambda _: dist.Normal(0.,1.), elementwise=True)), MCMCStep(AllVariables(), RW(gaussian_random_walk(0.05),elementwise=True))),
        MCMCSteps(MCMCStep(AllVariables(), RW(lambda _: dist.Normal(0.,1.), elementwise=True)), MCMCStep(AllVariables(), HMC(10,0.02))),
        # MCMCStep(AllVariables(), DHMC(10,0.01,0.02)),
        # MCMCStep(AllVariables(), HMC(10,0.02)),
        n_chains=n_chains,
        collect_inference_info=True,
        progress_bar=True,
        return_map=lambda x: (x.position, x.log_prob)
    )
    init_positions = broadcast_jaxtree(slp.decision_representative, (mcmc_obj.n_chains,))
    init_positions, init_lps = jax.vmap(slp.generate, in_axes=(0,None))(jax.random.split(jax.random.PRNGKey(0), mcmc_obj.n_chains), dict())
    print(init_positions["period"])
    print(init_lps)


    last_state, (positions, lp) = mcmc_obj.run(jax.random.PRNGKey(0), StackedTrace(init_positions, mcmc_obj.n_chains), n_samples_per_chain=n_samples_per_chain)
    print(summarise_mcmc_infos(last_state.infos, n_samples_per_chain))
    plt.hist(positions["lengthscale"].reshape(-1), bins=100, density=True, alpha=0.5, label="lengthscale")
    plt.hist(positions["amplitude"].reshape(-1), bins=100, density=True, alpha=0.5, label="amplitude")
    plt.hist(positions["period"].reshape(-1), bins=100, density=True, alpha=0.5, label="period")
    if "noise" in positions:
        plt.hist(positions["noise"].reshape(-1), bins=100, density=True, alpha=0.5, label="noise")
    plt.legend()
    plt.show()

    for i in range(mcmc_obj.n_chains):
        plt.figure()
        plt.plot(positions["lengthscale"][:,i], label="lengthscale")
        plt.plot(positions["amplitude"][:,i], label="amplitude")
        plt.plot(positions["period"][:,i], label="period")
        if "noise" in positions:
            plt.plot(positions["noise"][:,i], label="noise")
        plt.legend()
    plt.show()

    positions = StackedTraces(positions, n_samples_per_chain, mcmc_obj.n_chains)
    amax = jnp.unravel_index( jnp.argmax(lp), lp.shape)
    print(amax)
    map_position = positions.get_selection(amax[0], amax[1])
    k, noise = get_kernel(map_position)
    xs_pred = jnp.linspace(0.,2.,500)
    post = k.posterior_predictive(xs, ys, noise+1e-5, xs_pred, noise+1e-5)
    plt.scatter(xs, ys)
    plt.plot(xs_pred, post.numpyro_base.mean)
    plt.show()


    for i in range(mcmc_obj.n_chains):
        try:
            plt.figure()
            chain = positions.get_chain(i)
            # map_position = chain.get_ix(int(jnp.argmax(lp[:,i])))
            map_position = chain.get_ix(-1)
            k, noise = get_kernel(map_position)
            print(i, k.pprint())
            xs_pred = jnp.linspace(0.,2.,500)
            post = k.posterior_predictive(xs, ys, noise+1e-5, xs_pred, noise+1e-5)
            plt.scatter(xs, ys)
            plt.plot(xs_pred, post.numpyro_base.mean)
        except Exception as e:
            print(e)
    plt.show()


if False:
    z = jnp.linspace(-3,3,100)

    @jax.jit
    def I(lpa):
        l, p, a = lpa
        return slp.log_prob({"lengthscale": l, "period": p, "amplitude": a})

    z1, z2, z3 = jnp.meshgrid(z,z,z)
    print(z1.shape)


    lp = jax.lax.map(I, (z1.reshape(-1), z2.reshape(-1), z3.reshape(-1)), batch_size=1_000)
    log_Z = jax.scipy.special.logsumexp(lp) + 3*jnp.log(z[1]-z[0])
    print(log_Z)


log_Z, ESS, _ = estimate_log_Z_for_SLP_from_prior(slp, 100_000, jax.random.PRNGKey(0))
print(log_Z, ESS)

n_particles = 10

# rejuvination_regime = MCMCSteps(MCMCStep(AllVariables(), RW(lambda _: dist.Normal(0.,1.), elementwise=True)), MCMCStep(AllVariables(), RW(gaussian_random_walk(0.05),elementwise=True)))
# rejuvination_regime = MCMCStep(AllVariables(), HMC(10,0.02))
rejuvination_regime = MCMCSteps(
    MCMCStep(AllVariables(), RW(lambda x: dist.Normal(jax.lax.zeros_like_array(x),1.), elementwise=True)),
    MCMCStep(AllVariables(), HMC(10,0.02))
)

# tempering_schedule = tempering_schedule_from_sigmoid(jnp.linspace(-5,5,10))
# data_annealing_schedule = None

tempering_schedule = None
data_annealing_schedule = data_annealing_schedule_from_range({"obs": range(0,len(ys),13)})

smc_obj = SMC(
    slp,
    n_particles,
    tempering_schedule,
    data_annealing_schedule,
    ReweightingType.BootstrapStaticPrior,
    MultinomialResampling(ResampleType.Always, ResampleTime.Never),
    rejuvination_regime,
    75,
    collect_inference_info=True,
    progress_bar=True
)
# plt.plot(smc_obj.tempereture_schedule.temperature)
# plt.show()

# mcmc = MCMC(
#     slp,
#     MCMCStep(AllVariables(), HMC(10,0.2)),
#     # smc_obj.rejuvination_regime,
#     n_chains=smc_obj.n_particles,
#     # reuse_kernel=smc_obj.rejuvination_kernel,
#     # reuse_kernel_init_carry_stat_names=smc_obj.rejuvination_kernel_init_carry_stat_names,
#     data_annealing = smc_obj.data_annealing_schedule.prior_mask() if smc_obj.data_annealing_schedule is not None else dict(),
#     temperature = jnp.array(0.,float),
#     collect_inference_info=smc_obj.collect_inference_info,
#     progress_bar=True
# )
# init_positions = StackedTrace(broadcast_jaxtree(slp.decision_representative, (mcmc.n_chains,)), mcmc.n_chains)

# last_state, _ = mcmc.run(jax.random.PRNGKey(0), init_positions, n_samples_per_chain=10)
# print(jnp.sum(jnp.abs(last_state.log_prob - jax.vmap(slp.log_prior)(last_state.position))))

# init_particles = last_state.position

init_particles, _ = jax.vmap(slp.generate, in_axes=(0,None))(jax.random.split(jax.random.PRNGKey(0), smc_obj.n_particles), dict())

last_state, ess = smc_obj.run(jax.random.PRNGKey(0), StackedTrace(init_particles, n_particles))
print(get_log_Z_ESS(last_state.log_particle_weights))
# plt.plot(ess)
# plt.show()
print(summarise_mcmc_infos(last_state.mcmc_infos, smc_obj.n_steps*smc_obj.rejuvination_attempts))


particles = StackedTrace(last_state.particles, smc_obj.n_particles)
# xs_pred = jnp.linspace(0.,1.25,500)
xs_pred = jnp.sort(jnp.hstack((xs, jnp.linspace(1.,1.5,100))))
plt.figure()
plt.scatter(xs, ys)
for i in range(smc_obj.n_particles):
    particle = particles.get_ix(i)
    k, noise = get_kernel(particle)
    post = k.posterior_predictive(xs, ys, noise+1e-5, xs_pred, noise+1e-5)
    print(i, k.pprint(), last_state.log_particle_weights[i], last_state.ta_log_prob[i], noise)
    plt.plot(xs_pred, post.numpyro_base.mean, color="black")
    plt.fill_between(xs_pred, *mvnormal_quantiles(post, [0.025, 0.975]), alpha=0.1, color="tab:blue")

plt.show()