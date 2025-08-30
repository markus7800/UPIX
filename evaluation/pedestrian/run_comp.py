import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("--show_plots", action="store_true")
parser.add_argument("--show_scatter", action="store_true")
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
from dccxjax.infer import LogWeightEstimateFromPrior, estimate_log_Z_for_SLP_from_prior
from dccxjax.infer import AllVariables, DHMC

class DCCConfig(MCMCDCC[T]):
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        regime = MCMCStep(AllVariables(), DHMC(50, 0.05, 0.15, unconstrained=False))
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


if __name__ == "__main__":
    return_keys = ["start"]
    if args.show_scatter:
        return_keys += ["step_1", "step_2", "step_3"]

    dcc_obj = DCCConfig(m, verbose=2,
                parallelisation=get_parallelisation_config(args),
                init_n_samples=250,
                init_estimate_weight_n_samples=2**20, # ~10**6
                mcmc_n_chains=16,
                mcmc_n_samples_per_chain=25_000,
                estimate_weight_n_samples=2**23, # ~10**7
                max_iterations=1,
                mcmc_collect_for_all_traces=True,
                mcmc_optimise_memory_with_early_return_map=True,
                return_map=lambda trace: {key: trace[key] for key in return_keys if key in trace}
    )

    result, timings = timed(dcc_obj.run)(jax.random.key(0))
    result.pprint(sortkey="slp")


    gt_xs = jnp.load("evaluation/pedestrian/gt_xs-100.npy")
    gt_cdf = jnp.load("evaluation/pedestrian/gt_cdf-100-1_000_000_000_000.npy")
    gt_pdf = jnp.load("evaluation/pedestrian/gt_pdf_est-100-1_000_000_000_000.npy")


    if args.show_plots:
        slp = result.get_slp(lambda slp: find_t_max(slp) == 2)
        assert slp is not None
        traces, _ = result.get_samples_for_slp(slp).unstack().get()

        if "step_2" in traces:
            plt.scatter(traces["step_1"], traces["step_2"], alpha=0.1, s=1)
            plt.xlabel("step_1")
            plt.ylabel("step_2")
            plt.title(slp.formatted())

        slp = result.get_slp(lambda slp: find_t_max(slp) == 3)
        assert slp is not None
        traces, _ = result.get_samples_for_slp(slp).unstack().get()

        if "step_3" in traces:
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
        if "step_2" in traces:
            plot_histogram_by_slp(result, "step_1")
            plot_histogram_by_slp(result, "step_2")
        plt.show()

        plot_histogram(result, "start")
        fig = plt.gcf()
        ax = fig.axes[0]
        ax.plot(gt_xs, gt_pdf)
        plt.savefig("evaluation/pedestrian/result_dccxjax.pdf")
        plt.show()


    start_weighted_samples, _ = result.get_samples_for_address("start", sample_ixs=slice(1000,None)) # burn-in
    assert start_weighted_samples is not None
    start_samples, start_weights = start_weighted_samples.get()

    @jax.jit
    def cdf_estimate(sample_points, sample_weights: jax.Array, qs):
        def _cdf_estimate(q):
            return jnp.where(sample_points < q, sample_weights, jax.numpy.zeros_like(sample_weights)).sum()
        return jax.lax.map(_cdf_estimate, qs)

    cdf_est = cdf_estimate(start_samples, start_weights, gt_xs)
    W1_distance = jnp.trapezoid(jnp.abs(cdf_est - gt_cdf)) # wasserstein distance
    infty_distance = jnp.max(jnp.abs(cdf_est - gt_cdf))
    title = f"W1 = {W1_distance.item():.4g}, L_inf = {infty_distance.item():.4g}"
    print(title)

    plt.plot(gt_xs, jnp.abs(cdf_est - gt_cdf))
    plt.title(title)
    plt.show()
    
    
    workload = {
        "n_chains": dcc_obj.mcmc_n_chains,
        "n_samples_per_chain": dcc_obj.mcmc_n_samples_per_chain,
    }

    result_metrics = {
        "W1": W1_distance.item(),
        "L_inf": infty_distance.item()
    }
        
    json_result = {
        "workload": workload,
        "timings": timings,
        "dcc_timings": dcc_obj.get_timings(),
        "result_metrics": result_metrics,
        "args": args.__dict__,
        "pconfig": dcc_obj.pconfig.__dict__,
        "environment_info":  get_environment_info()
    }
    
    if not args.no_save:
        write_json_result(json_result, "experiments", "pedestrian", "scale", prefix=f"nchains_{dcc_obj.mcmc_n_chains:07d}_")
