import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("--show_plots", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)

from upix.core import *
from upix.viz import *
from setup_parallelisation import get_parallelisation_config

from gp_smc import *
from smc_plots import plot_results

AutoGPConfig()
# RecheiltConfig()

if __name__ == "__main__":
    
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    m.set_equivalence_map(equivalence_map)

    smc_dcc_obj = SMCDCCConfig(m, verbose=2,
        init_n_samples=1_000,
        smc_rejuvination_attempts=8,
        smc_n_particles=100,
        smc_collect_inference_info=True,
        max_iterations = 5,
        n_lmh_update_samples = 250,
        max_active_slps = 5,
        max_new_active_slps = 5,
        one_inference_run_per_slp = True,
        parallelisation = get_parallelisation_config(args),
        disable_progress=args.no_progress
    )

    result, timings = timed(smc_dcc_obj.run)(jax.random.key(0))
    result.pprint()

    if args.show_plots:
        plot_results(m, result)
