#%%
import sys
sys.path.append("evaluation")
from parse_args import parse_args_and_setup
args = parse_args_and_setup()

from dccxjax.all import *
import jax
import jax.numpy as jnp
import dccxjax.distributions as dist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict
from time import time
from setup_parallelisation import get_parallelisation_config

import logging
setup_logging(logging.WARNING)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)



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


def pedestrian_rec_walk(t: int, position: FloatArrayLike, distance: FloatArrayLike) -> FloatArrayLike:
    if (position > 0) & (distance < 10):
        step = sample(f"step_{t}", dist.Uniform(-1.,1.))
        position += step
        distance += jax.lax.abs(step)
        return pedestrian_rec_walk(t+1, position, distance)
    else:
        return distance

def pedestrian_recursive():
    start = sample("start", dist.Uniform(0.,3.))
    distance = pedestrian_rec_walk(1, start, 0.)
    sample("obs", dist.Normal(distance, 0.1), observed=1.1)
    return start


# m = model(pedestrian_recursive)()
m = model(pedestrian)()

def find_t_max(slp: SLP):
    t_max = 0
    for addr in slp.decision_representative.keys():
        if addr.startswith("step_"):
            t_max = max(t_max, int(addr[5:]))
    return t_max
def formatter(slp: SLP):
    t_max = find_t_max(slp)
    return f"#steps={t_max}"
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_t_max)

class DCCConfig(MCMCDCC[T]):
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        # regime = MCMCSteps(
        #     MCMCStep(PrefixSelector("step"), RandomWalk(lambda x: dist.TwoSidedTruncatedDistribution(dist.Normal(x, 0.2), -1.,1.), sparse_numvar=2)),
        #     MCMCStep(SingleVariable("start"), RandomWalk(lambda x: dist.TwoSidedTruncatedDistribution(dist.Normal(x, 0.2), 0., 3.)))
        # )
        # regime = MCMCStep(AllVariables(), HMC(10, 0.05, unconstrained=True))
        # regime = MCMCStep(AllVariables(), DHMC(10, 0.05, 0.15, unconstrained=False)) # this is very good W1 = 0.01796, L_inf = 0.0008
        regime = MCMCStep(AllVariables(), DHMC(50, 0.05, 0.15, unconstrained=False)) # W1 = 0.01503, L_inf = 0.001072 for comparison takes ~90s for 25_000 * 10 * 6 samples
        tqdm.write(pprint_mcmc_regime(regime, slp))
        return regime
    
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        _active_slps: List[SLP] = []
        super().initialise_active_slps(_active_slps, inactive_slps, rng_key)
        # we assume that we know that with increasing steps eventually we have unlikely SLPs
        _active_slps.sort(key=self.model.slp_sort_key)
        log_Z_max = -jnp.inf
        for slp in _active_slps:
            rng_key, estimate_key = jax.random.split(rng_key)
            estimate_weight_n_samples: int = self.config["init_estimate_weight_n_samples"]
            log_Z, ESS, frac_in_support = estimate_log_Z_for_SLP_from_prior(slp, estimate_weight_n_samples, estimate_key)
            log_Z_max = jax.lax.max(log_Z, log_Z_max)
            if log_Z - log_Z_max > jnp.log(0.001):
                self.add_to_log_weight_estimates(slp, LogWeightEstimateFromPrior(log_Z, ESS, frac_in_support, estimate_weight_n_samples))
                active_slps.append(slp)
                tqdm.write(f"Make SLP {slp.formatted()} active (log_Z={log_Z.item():.4f}).")
            else:
                # we are so far past maximum likely SLP that we can safely break
                break
    
    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        if self.iteration_counter == self.max_iterations:
            inactive_slps.extend(active_slps)
            active_slps.clear()
        else:
            assert not self.mcmc_optimise_memory_with_early_return_map


dcc_obj = DCCConfig(m, verbose=2,
              parallelisation=get_parallelisation_config(args),
              init_n_samples=250,
              init_estimate_weight_n_samples=1_000_000,
              mcmc_n_chains=10,
              mcmc_n_samples_per_chain=25_000,
              estimate_weight_n_samples=10_000_000,
              max_iterations=1,
              mcmc_collect_for_all_traces=True,
              mcmc_optimise_memory_with_early_return_map=True,
              return_map=lambda trace: {"start": trace["start"]}
)

result = timed(dcc_obj.run)(jax.random.PRNGKey(0))
result.pprint(sortkey="slp")

exit()


gt_xs = jnp.load("evaluation/pedestrian/gt_xs.npy")
gt_cdf = jnp.load("evaluation/pedestrian/gt_cdf.npy")
gt_pdf = jnp.load("evaluation/pedestrian/gt_pdf.npy")

show_plots = True
if show_plots:
    slp = result.get_slp(lambda slp: find_t_max(slp) == 2)
    assert slp is not None
    traces, _ = result.get_samples_for_slp(slp).unstack().get()

    if "step_1" in traces:
        plt.scatter(traces["step_1"], traces["step_2"], alpha=0.1, s=1)
        plt.xlabel("step_1")
        plt.ylabel("step_2")
        plt.title(slp.formatted())

    slp = result.get_slp(lambda slp: find_t_max(slp) == 3)
    assert slp is not None
    traces, _ = result.get_samples_for_slp(slp).unstack().get()

    if "step_1" in traces:
        plt.figure()
        plt.scatter(traces["step_1"], traces["step_2"], alpha=0.1, s=1)
        plt.xlabel("step_1")
        plt.ylabel("step_2")
        plt.title(slp.formatted())
        plt.figure()
        plt.scatter(traces["step_2"], traces["step_3"], alpha=0.1, s=1)
        plt.xlabel("step_2")
        plt.ylabel("step_3")
        plt.title(slp.formatted())
        plt.figure()
        plt.scatter(traces["step_1"], traces["step_3"], alpha=0.1, s=1)
        plt.xlabel("step_1")
        plt.ylabel("step_3")
        plt.title(slp.formatted())
        plt.show()


    plot_histogram_by_slp(result, "start")
    if "step_1" in traces:
        plot_histogram_by_slp(result, "step_1")
        plot_histogram_by_slp(result, "step_2")
    plt.show()

    plot_histogram(result, "start")
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.plot(gt_xs, gt_pdf)
    plt.savefig("evaluation/pedestrian/result_dccxjax.pdf")
    plt.show()

#%%
start_weighted_samples, _ = result.get_samples_for_address("start", sample_ixs=slice(1000,None)) # burn-in
assert start_weighted_samples is not None
start_samples, start_weights = start_weighted_samples.get()

#%%
@jax.jit
def cdf_estimate(sample_points, sample_weights: jax.Array, qs):
    def _cdf_estimate(q):
        return jnp.where(sample_points < q, sample_weights, jax.lax.zeros_like_array(sample_weights)).sum()
    return jax.lax.map(_cdf_estimate, qs)

cdf_est = cdf_estimate(start_samples, start_weights, gt_xs)
W1_distance = jnp.trapezoid(jnp.abs(cdf_est - gt_cdf)) # wasserstein distance
infty_distance = jnp.max(jnp.abs(cdf_est - gt_cdf))
title = f"W1 = {W1_distance.item():.4g}, L_inf = {infty_distance.item():.4g}"
print(title)

plt.plot(gt_xs, jnp.abs(cdf_est - gt_cdf))
plt.title(title)
plt.show()
# %%
