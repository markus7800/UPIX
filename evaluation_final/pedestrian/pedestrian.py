import sys
import os
sys.path.insert(0, ".")
if len(sys.argv) > 1:
    if sys.argv[1].endswith ("cpu"):
        print("Force run on CPU.")
        os.environ["JAX_PLATFORMS"] = "cpu"


from dccxjax import *
import jax
import jax.numpy as jnp
import dccxjax.distributions as dist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List
from time import time
from dccxjax.infer.dcc2 import *

import logging
setup_logging(logging.WARNING)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)

t0 = time()

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


class DCCConfig(MCMCDCC[DCC_COLLECT_TYPE]):
    def get_MCMC_inference_regime(self, slp: SLP) -> InferenceRegime:
        return Gibbs(
            InferenceStep(PrefixSelector("step"), RandomWalk(lambda x: dist.TwoSidedTruncatedDistribution(dist.Normal(x, 0.2), -1.,1.), sparse_numvar=2)),
            InferenceStep(SingleVariable("start"), RandomWalk(lambda x: dist.TwoSidedTruncatedDistribution(dist.Normal(x, 0.2), 0., 3.)))
        )
    def initialise_active_slps(self, active_slps: List[SLP], rng_key: jax.Array):
        _active_slps: List[SLP] = []
        super().initialise_active_slps(_active_slps, rng_key)
        # we assume that we know that with increasing steps eventually we have unlikely SLPs
        _active_slps.sort(key=self.model.slp_sort_key)
        # _active_slps.sort(key=lambda slp: slp.sort_key())
        log_Z_max = -jnp.inf
        for slp in _active_slps:
            # if find_t_max(slp) <= 6:
            #     active_slps.append(slp)

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


dcc_obj = DCCConfig(m, verbose=2,
              init_n_samples=250,
              init_estimate_weight_n_samples=1_000_000,
              mcmc_n_chains=100,
              mcmc_n_samples_per_chain=100_000,
              mcmc_collect_for_all_traces=True,
              estimate_weight_n_samples=100_000_000,
              return_map=lambda trace: {"start": trace["start"]})

t0 = time()

result = dcc_obj.run(jax.random.PRNGKey(0))
result.pprint()

t1 = time()

print(f"Total time: {t1-t0:.3f}s")
comp_time = compilation_time_tracker.get_total_compilation_time_secs()
print(f"Total compilation time: {comp_time:.3f}s ({comp_time / (t1 - t0) * 100:.2f}%)")

# plot_histogram(result, "start")
# plot_trace(result, "start")
plot_histogram_by_slp(result, "start")
plt.show()


gt_xs = jnp.load("evaluation_final/pedestrian/gt_xs.npy")
gt_cdf = jnp.load("evaluation_final/pedestrian/gt_cdf.npy")
gt_pdf = jnp.load("evaluation_final/pedestrian/gt_pdf.npy")

plot_histogram(result, "start")
fig = plt.gcf()
ax = fig.axes[0]
ax.plot(gt_xs, gt_pdf)
plt.show()

start_samples, start_weights, _ = result.get_samples_for_address("start")
assert start_samples is not None and start_weights is not None

@jax.jit
def cdf_estimate(sample_points, sample_weights: jax.Array, qs):
    def _cdf_estimate(q):
        return jnp.where(sample_points < q, sample_weights, jax.lax.zeros_like_array(sample_weights)).sum()
    return jax.lax.map(_cdf_estimate, qs)

cdf_est = cdf_estimate(start_samples, start_weights, gt_xs)
W1_distance = jnp.trapezoid(jnp.abs(cdf_est - gt_cdf)) # wasserstein distance
infty_distance = jnp.max(jnp.abs(cdf_est - gt_cdf))

plt.plot(gt_xs, jnp.abs(cdf_est - gt_cdf))
plt.title(f"W1 = {W1_distance.item():.4g}, L^\\infty = {infty_distance.item():.4g}")
plt.show()