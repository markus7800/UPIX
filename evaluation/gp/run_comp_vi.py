import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("-L", type=int, default=1)
parser.add_argument("-n_runs", type=int, default=1)
parser.add_argument("-sh_iterations", type=int, default=1_000_000)
parser.add_argument("--show_plots", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)

from upix.core import *
from upix.viz import *
from setup_parallelisation import get_parallelisation_config

from gp_vi import *
from upix.infer.variational_inference.optimizers import Adagrad, SGD, Adam
from vi_utils import plot_results, compute_lppd, save_results

# AutoGPConfig()
# xs, xs_val, ys, ys_val, rescale_x, rescale_y = get_data_autogp()
RecheiltConfig()
xs, xs_val, ys, ys_val, rescale_x, rescale_y = get_data_sdvi()

if __name__ == "__main__":
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    # m.set_equivalence_map(equivalence_map)
    
    vi_dcc_obj = VIConfig(m, verbose=2,
        init_n_samples=1_000,
        advi_L=args.L,
        advi_n_runs=args.n_runs, # 8 would work better
        advi_optimizer=Adam(0.005),
        elbo_estimate_n_samples=100,
        successive_halving=SuccessiveHalving(args.sh_iterations, 10),
        parallelisation = get_parallelisation_config(args),
        disable_progress=args.no_progress
    )

    result, timings = timed(vi_dcc_obj.run)(jax.random.key(args.seed))
    result.pprint()

    pells_lppds = [compute_lppd(result, xs, ys, xs_val, ys_val, 1000, i) for i in range(5)]
    pell, lppd = pells_lppds[0] # save first independent of lppd_reps
    
    pells = jnp.array([pell for pell, _ in pells_lppds], float)
    lppds = jnp.array([lppd for _, lppd in pells_lppds], float)
    pell, pell_std = float(pells[0]), float(pells.std()) # save first one independent of std
    lppd, lppd_std = float(lppds[0]), float(lppds.std())
    print(f"pell: {float(pells.mean())} +/- {pell_std}", f"lppd: {float(lppds.mean())} +/- {lppd_std}")
    
    if args.show_plots:
        plot_results(result, xs, ys, xs_val, ys_val)
    
    if not args.no_save:
        save_results(args, result, vi_dcc_obj, timings, pell, pell_std, lppd, lppd_std, "comp")

