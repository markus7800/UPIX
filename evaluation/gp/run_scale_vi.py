import sys
from typing import List

sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
if __name__ == "__main__":
    parser.add_argument("n_slps", help="number of slps to evaluate", type=int)
    parser.add_argument("n_runs", help="number of parallel ADVI runs", type=int)
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
from vi_plots import plot_results

from enumerate_slps import find_active_slps_through_enumeration

RecheiltConfig()

class VIConfig2(VIConfig):
    def __init__(self, model: Model, *ignore, verbose=0, **config_kwargs) -> None:
        VIDCC.__init__(self, model, *ignore, verbose=verbose, **config_kwargs)

    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        find_active_slps_through_enumeration(NODE_CONFIG.N_LEAF_NODE_TYPES, active_slps, rng_key, args.n_slps, self.model, max_n_leaf=self.config["slp_max_n_leaf"])
    
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
        advi_n_runs=args.n_runs,
        advi_optimizer=Adam(0.005),
        elbo_estimate_n_samples=100,
        parallelisation = get_parallelisation_config(args),
        disable_progress=args.no_progress,
        slp_max_n_leaf = 4,
    )

    result, timings = timed(vi_dcc_obj.run)(jax.random.key(0))
    result.pprint()
    
    if args.show_plots:
        plot_results(result)
        
        
    K = int(args.n_runs) * int(args.L)
    workload = {
        "K": K,
        "L": args.L,
        "n_runs": args.n_runs,
        "n_iter": args.n_iter,
        "n_slps": len(result.get_slps()),
        "config": NODE_CONFIG.NAME
    }

    result_metrics = {
        "result_str": result.sprint(sortkey="slp"),
        "pmap_check": str(check_pmap())
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
        prefix = f"K_{K:07d}_nruns_{args.n_runs}_L_{args.L}_nslps_{len(result.get_slps())}_niter_{args.n_iter}_"
        write_json_result(json_result, "gp", "vi", "scale", prefix=prefix)


