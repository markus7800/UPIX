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

from upix.core import *
from upix.viz import *
from setup_parallelisation import get_parallelisation_config

from gp_vi import *
from upix.infer.variational_inference.optimizers import Adagrad, SGD, Adam
from vi_utils import plot_results, compute_lppd, save_results

from enumerate_slps import find_active_slps_through_enumeration

# AutoGPConfig()
# xs, xs_val, ys, ys_val, rescale_x, rescale_y = get_data_autogp()
RecheiltConfig()
xs, xs_val, ys, ys_val, rescale_x, rescale_y = get_data_sdvi()

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

    result, timings = timed(vi_dcc_obj.run)(jax.random.key(args.seed))
    result.pprint()
    
    pell, lppd = compute_lppd(result, xs, ys, xs_val, ys_val, 100)
    print("pell:", pell, "lppd:", lppd)
    if args.show_plots:
        plot_results(result, xs, ys, xs_val, ys_val)

    if not args.no_save:
        save_results(args, result, vi_dcc_obj, timings, pell, lppd, "scale")