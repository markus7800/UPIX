import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("--show_plots", action="store_true")
parser.add_argument("--show_scatter", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)

from upix.core import *
from upix.viz import *
from setup_parallelisation import get_parallelisation_config

import logging
setup_logging(logging.WARN)

from pedestrian import *

# m = pedestrian_recursive()
m = pedestrian()
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_t_max)


from upix.infer import MCMCDCC, T, MCMCRegime, MCMCStep, MCMCSteps, InferenceResult, LogWeightEstimate
from upix.infer import LogWeightEstimateFromPrior, estimate_log_Z_for_SLP_from_prior
from upix.infer import AllVariables, DHMC, Variables, MH, RW

class DCCConfig(MCMCDCC[T]):
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        regime = MCMCSteps(
            MCMCStep(Variables("start"), RW(lambda x: dist.Uniform(jnp.zeros_like(x),3))),
            MCMCStep(Variables(r"step_\d+"), DHMC(50, 0.05, 0.15)),
        )
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
            log_Z, ESS, frac_in_support = estimate_log_Z_for_SLP_from_prior(slp, estimate_weight_n_samples, estimate_key, self.pconfig)
            log_Z_max = jax.lax.max(log_Z, log_Z_max)
            if log_Z - log_Z_max > jnp.log(0.1):
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


if __name__ == "__main__":
    return_keys = ["start"]
    if args.show_scatter:
        return_keys += ["step_1", "step_2", "step_3"]

    dcc_obj = DCCConfig(m, verbose=2,
                parallelisation=get_parallelisation_config(args),
                init_n_samples=250,
                init_estimate_weight_n_samples=2**20, # ~10**6
                mcmc_n_chains=8,
                mcmc_n_samples_per_chain=25_000,
                estimate_weight_n_samples=2**23, # ~10**7
                max_iterations=1,
                mcmc_collect_for_all_traces=True,
                mcmc_optimise_memory_with_early_return_map=True,
                return_map=lambda trace: {key: trace[key] for key in return_keys if key in trace},
                disable_progress=args.no_progress
    )

    result, timings = timed(dcc_obj.run)(jax.random.key(0))
    result.pprint(sortkey="slp")


    gt_xs = jnp.load("evaluation/pedestrian/gt_xs-100.npy")
    gt_cdf = jnp.load("evaluation/pedestrian/gt_cdf-100-1_000_000_000_000.npy")
    gt_pdf = jnp.load("evaluation/pedestrian/gt_pdf_est-100-1_000_000_000_000.npy")


    plot_histogram_by_slp(result, "start")
    
    
    start_weighted_samples, _ = result.get_samples_for_address("start", sample_ixs=slice(1000,None)) # burn-in
    assert start_weighted_samples is not None
    start_samples, start_weights = start_weighted_samples.unstack().get()
    
    axes = plt.gcf().axes
    print(axes)
    axes[-2].plot(gt_xs, gt_pdf, label="ground truth", linestyle="dashed")
    axes[-2].legend()
    plt.show()

