import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("n_slps", help="number of slps to evaluate", type=int)
parser.add_argument("n_chains", help="number of chains to run", type=int)
parser.add_argument("n_samples_per_chain", help="number of sampler per chain to run", type=int)
parser.add_argument("--show_plots", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)

from dccxjax.core import *
from dccxjax.viz import *
from setup_parallelisation import get_parallelisation_config

import logging
setup_logging(logging.WARN)

from pedestrian import *

# m = pedestrian_recursive()
m = pedestrian()
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_t_max)


from dccxjax.infer import MCMCDCC, T, MCMCRegime, MCMCStep, InferenceResult, LogWeightEstimate
from dccxjax.infer import AllVariables, DHMC

class DCCConfig(MCMCDCC[T]):
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        regime = MCMCStep(AllVariables(), DHMC(50, 0.05, 0.15, unconstrained=False))
        return regime
    
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        _active_slps: List[SLP] = []
        super().initialise_active_slps(_active_slps, inactive_slps, rng_key)
        _active_slps.sort(key=self.model.slp_sort_key)
        for k, slp in enumerate(_active_slps):
            if k < args.n_slps:
                active_slps.append(slp)
                tqdm.write(f"Make SLP {slp.formatted()} active.")
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
              init_estimate_weight_n_samples=2**20, # ~10**6
              mcmc_n_chains=args.n_chains,
              mcmc_n_samples_per_chain=args.n_samples_per_chain,
              estimate_weight_n_samples=2**23, # ~10**7
              max_iterations=1,
              mcmc_collect_for_all_traces=False,
              mcmc_optimise_memory_with_early_return_map=True,
              return_map=lambda trace: {"start": trace["start"]}
)

result = timed(dcc_obj.run)(jax.random.key(0))
result.pprint(sortkey="slp")


gt_xs = jnp.load("evaluation/pedestrian/gt_xs.npy")
gt_cdf = jnp.load("evaluation/pedestrian/gt_cdf.npy")
gt_pdf = jnp.load("evaluation/pedestrian/gt_pdf.npy")




if args.show_plots:
    plot_histogram(result, "start")
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.plot(gt_xs, gt_pdf)
    plt.savefig("evaluation/pedestrian/result_dccxjax.pdf")
    plt.show()


start_weighted_samples, _ = result.get_samples_for_address("start") 
assert start_weighted_samples is not None
start_samples, start_weights = start_weighted_samples.get()

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

# (venv) markus@Markuss-MBP-14 DCCxJAX % python3 evaluation/pedestrian/run_scale.py sequential smap_local 8 16384 1000 -host_device_count 8 --show_plots