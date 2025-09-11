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
from dccxjax.parallelisation import VectorisationType

from gp_smc_sh import *
from smc_plots import plot_results


AutoGPConfig()
# RecheiltConfig()

if __name__ == "__main__":
    
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    m.set_equivalence_map(equivalence_map)

    smc_dcc_obj = SMCDCCConfig2(m, verbose=2,
        smc_collect_inference_info=True,
        init_n_samples=1_000,
        successive_halving=SuccessiveHalving(10_000, 10),
        parallelisation = get_parallelisation_config(args),
        round_n_particles_to_multiple = jax.device_count(),
        disable_progress=args.no_progress
    )
    # assert smc_dcc_obj.pconfig.vectorisation == VectorisationType.LocalVMAP

    result, timings = timed(smc_dcc_obj.run)(jax.random.key(0))
    result.pprint()
    
    if args.show_plots:
        plot_results(m, result)


# ((1Per * 1SqExp) + 1Poly): StackedSampleValues(dict, 1 x 530) with prob=0.000000, log_Z=144.877136
# ((1Per * 1SqExp) + 1SqExp): StackedSampleValues(dict, 1 x 530) with prob=0.000000, log_Z=144.914948
# ((1Per * 1RQ) * 1Poly): StackedSampleValues(dict, 1 x 530) with prob=0.000017, log_Z=153.339279
# ((1Per + 1RQ) * 1Poly): StackedSampleValues(dict, 1 x 530) with prob=0.000021, log_Z=153.565842
# ((1Per * 1SqExp) + 1RQ): StackedSampleValues(dict, 1 x 330) with prob=0.000341, log_Z=156.338913
# ((1Per * 1Poly) + 1RQ): StackedSampleValues(dict, 1 x 530) with prob=0.999619, log_Z=164.322586