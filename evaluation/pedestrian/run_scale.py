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

    def get_initial_positions(self, slp: SLP, rng_key: PRNGKey) -> StackedTrace:
        # return super().get_initial_positions(slp, rng_key)
        
        # make initial positions more diverse
        
        N_STEPS = find_t_max(slp)
        
        # returns random numbers us such that for cs = start + cumsum(us)
        # we have (cs[:-1] > 0).all() and cs[-1] = stop
        def rw(seed: PRNGKey, start: FloatArray, stop: FloatArray):
            def step(carry, key):
                s, i = carry
                # force s to be > 1e-5 for all but last steps, for which we forst s > stop
                minval = jax.lax.max(jnp.array(-1,float), jax.lax.select(i == 1, -s + stop, -s + 1e-5))
                # force s small enough such that next steps can push final s to stop < 0
                maxval = jax.lax.min(jnp.array(1,float), (i-1)*(1-1e-5) - s + stop)
                u = jax.random.uniform(key, minval=minval, maxval=maxval)
                return (s + u, i-1), u
            _, us = jax.lax.scan(step, (start, N_STEPS), jax.random.split(seed, N_STEPS))
            return us
        
        @jax.jit
        @jax.vmap
        def get_positions(key: PRNGKey):
            key1, key2, key3 = jax.random.split(key,3)
            # draw start positions
            start = jax.random.uniform(key1, minval=0, maxval=jnp.minimum(N_STEPS,3))
            steps = rw(key2, start, jnp.array(-0.1,float))
            
            # draw distance target
            distance = 0.1 * jax.random.normal(key3) + 1.1
            
            # scale random walk to match distance
            scale = jnp.minimum(distance / jnp.sum(jnp.abs(steps)), 1)
            return start * scale, steps * scale
            
        start, steps = get_positions(jax.random.split(rng_key, self.mcmc_n_chains))
        traces = {
            "start": start
        }
        for t in range(N_STEPS):
            traces[f"step_{t+1}"] = steps[:,t]
            
        # check if all initial positions are in SLP support
        # _, _, pcs = jax.vmap(slp._log_prior_likeli_pathcond, in_axes=(0,None))(traces, dict())
        # tqdm.write(f"Mean pc {jnp.mean(pcs)}.")
            
        return StackedTrace(traces, self.mcmc_n_chains)
    

if __name__ == "__main__":
    
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
                return_map=lambda trace: {"start": trace["start"]},
                disable_progress=args.no_progress
    )

    result, timings = timed(dcc_obj.run)(jax.random.key(0))
    result.pprint(sortkey="slp")

    gt_xs = jnp.load("evaluation/pedestrian/gt_xs-100.npy")
    gt_cdf = jnp.load("evaluation/pedestrian/gt_cdf-100-1_000_000_000_000.npy")
    gt_pdf = jnp.load("evaluation/pedestrian/gt_pdf_est-100-1_000_000_000_000.npy")

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
            return jnp.where(sample_points < q, sample_weights, jax.numpy.zeros_like(sample_weights)).sum()
        return jax.lax.map(_cdf_estimate, qs)

    cdf_est = cdf_estimate(start_samples, start_weights, gt_xs)
    W1_distance = jnp.trapezoid(jnp.abs(cdf_est - gt_cdf)) # wasserstein distance
    infty_distance = jnp.max(jnp.abs(cdf_est - gt_cdf))
    title = f"W1 = {W1_distance.item():.4g}, L_inf = {infty_distance.item():.4g}"
    print(title)
    
    if args.show_plots:
        plt.plot(gt_xs, jnp.abs(cdf_est - gt_cdf))
        plt.title(title)
        plt.show()
        
    workload = {
        "n_chains": dcc_obj.mcmc_n_chains,
        "n_samples_per_chain": dcc_obj.mcmc_n_samples_per_chain,
        "n_slps": len(result.get_slps())
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
        prefix = f"nchains_{dcc_obj.mcmc_n_chains:07d}_nslps_{len(result.get_slps())}_niter_{dcc_obj.mcmc_n_samples_per_chain}_"
        write_json_result(json_result, "experiments", "pedestrian", "scale", prefix=prefix)

