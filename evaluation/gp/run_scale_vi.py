import sys
from typing import List

sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("n_slps", help="number of slps to evaluate", type=int)
parser.add_argument("L", help="number of samples to take per ADVI iteration", type=int)
parser.add_argument("n_iter", help="number of ADVI iterations", type=int)
parser.add_argument("--show_plots", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)

from dccxjax.core import *
from dccxjax.viz import *
from setup_parallelisation import get_parallelisation_config

from gp_vi import *
from dccxjax.infer.variational_inference.optimizers import Adagrad, SGD, Adam

from enumerate_slps import find_active_slps_through_enumeration

AutoGPConfig()

class VIConfig2(VIConfig):
    def __init__(self, model: Model, *ignore, verbose=0, **config_kwargs) -> None:
        VIDCC.__init__(self, model, *ignore, verbose=verbose, **config_kwargs)

    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        find_active_slps_through_enumeration(NODE_CONFIG.N_LEAF_NODE_TYPES, active_slps, rng_key, args.n_slps, self.model)
    
    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        inactive_slps.clear()
        inactive_slps.extend(active_slps)
        active_slps.clear()

if __name__ == "__main__":
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    m.set_equivalence_map(equivalence_map)
    
    vi_dcc_obj = VIConfig2(m, verbose=2,
        advi_n_iter = args.n_iter,
        advi_L=args.L,
        advi_optimizer=Adam(0.005),
        elbo_estimate_n_samples=100,
        parallelisation = get_parallelisation_config(args),
        disable_progress=args.no_progress
    )

    result, timings = timed(vi_dcc_obj.run)(jax.random.key(0))
    result.pprint()

    if args.show_plots:
        slp_weights = list(result.get_slp_weights().items())
        slp_weights.sort(key=lambda v: v[1])

        xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))
        
        for slp_ix in range(1,5+1):
            slp, weight = slp_weights[-slp_ix]
            print(slp.formatted(), weight)
            g = result.slp_guides[slp]
                
            n_posterior_samples = 1_000
                
            key = jax.random.key(0)
            posterior = Traces(g.sample(key, (n_posterior_samples,)), n_posterior_samples)
            
            samples = []
            for i in tqdm(range(n_posterior_samples), desc="Sample posterior of MAP SLP"):
                key, sample_key = jax.random.split(key)
                trace = posterior.get_ix(i)
                k = get_gp_kernel(trace)
                noise = transform_param("noise", trace["noise"]) + 1e-5
                mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)
                samples.append(mvn.sample(sample_key))

            samples = jnp.vstack(samples)
            m = jnp.mean(samples, axis=0)
            q025 = jnp.quantile(samples, 0.025, axis=0)
            q975 = jnp.quantile(samples, 0.975, axis=0)

            plt.figure()
            plt.title(f"{slp_ix}. {slp.formatted()}")
            plt.scatter(xs, ys)
            plt.scatter(xs_val, ys_val)
            plt.plot(xs_pred, m, color="black")
            plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
        plt.show()
        
        
    workload = {
        "L": args.L,
        "n_iter": args.n_iter,
        "n_slps": len(result.get_slps())
    }

    result_metrics = {
    }
        
    json_result = {
        "workload": workload,
        "timings": timings,
        "dcc_timings": vi_dcc_obj.get_timings(),
        "result_metrics": result_metrics,
        "args": args.__dict__,
        "pconfig": vi_dcc_obj.pconfig.__dict__,
        "environment_info": get_environment_info()
    }
    
    if not args.no_save:
        prefix = f"L_{args.L:07d}_nslps_{len(result.get_slps())}_niter_{args.n_iter}_"
        write_json_result(json_result, "gp", "vi", "scale", prefix=prefix)


