import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("--show_plots", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)

from dccxjax.core import *
from dccxjax.viz import *
from setup_parallelisation import get_parallelisation_config

from gp_vi import *
from dccxjax.infer.variational_inference.optimizers import Adagrad, SGD, Adam
from vi_plots import plot_results

RecheiltConfig()

if __name__ == "__main__":
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    # m.set_equivalence_map(equivalence_map)
    
    vi_dcc_obj = VIConfig(m, verbose=2,
        init_n_samples=1_000,
        advi_L=1,
        advi_n_runs=1, # 8 would work better
        advi_optimizer=Adam(0.005),
        elbo_estimate_n_samples=100,
        successive_halving=SuccessiveHalving(1_000_000, 10),
        parallelisation = get_parallelisation_config(args),
        disable_progress=args.no_progress
    )

    result, timings = timed(vi_dcc_obj.run)(jax.random.key(0))
    result.pprint()

    if args.show_plots:
        plot_results(result)