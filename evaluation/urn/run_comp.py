import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("--show_plots", action="store_true")
parser.add_argument("n_slps", type=int)
parser.add_argument("--jit_inf", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)

from upix.core import *
from upix.viz import *
from setup_parallelisation import get_parallelisation_config

from urn import *

import matplotlib.pyplot as plt

if __name__ == "__main__":

    obs = jnp.array([0,1] * 5)
    m = urn(obs, True)

    def _get_n(slp: SLP) -> int:
        return int(slp.decision_representative["N"].item())
    m.set_slp_formatter(lambda slp: f"N={_get_n(slp)}")
    m.set_slp_sort_key(_get_n)

    config = Config(m, verbose=2,
        parallelisation = get_parallelisation_config(args),
        jit_inference=args.jit_inf,
        N_max = args.n_slps
    )

    result, timings = timed(config.run)(jax.random.key(args.seed))
    result.pprint(sortkey="slp")

    gt = jnp.load("evaluation/urn/gt_ps.npy")
    log_Zs = jnp.array([log_Z for (slp, log_Z) in result.get_log_weights_sorted(sortkey="slp")])

    assert (jnp.array([slp.decision_representative["N"] for (slp, log_Z) in result.get_log_weights_sorted(sortkey="slp")]) == jnp.arange(1,len(log_Zs)+1)).all()

    ps = jnp.exp(log_Zs - jax.scipy.special.logsumexp(log_Zs))
    print(ps)
    print("vs")
    print(gt[:len(ps)])
    
    err = jnp.abs(jnp.hstack((ps, jnp.zeros((len(gt)-len(ps),)))) - gt)
    if args.show_plots:
        plt.plot(err)
        plt.show()
    l_inf_distance = jnp.max(err)
    print("Max err: ", l_inf_distance)
    
    workload = {
        "n_slps":  args.n_slps,
        "jit_inf": args.jit_inf,
        "seed": args.seed
    }
    
    result_metrics = {
        "L_inf": l_inf_distance.item()
    }
        
    json_result = {
        "workload": workload,
        "timings": timings,
        "dcc_timings": config.get_timings(),
        "result_metrics": result_metrics,
        "args": args.__dict__,
        "pconfig": config.pconfig.__dict__,
        "environment_info": get_environment_info()
    }
    
    prefix = f"nslps_{len(result.get_slps())}_jitinf_{args.jit_inf}_"
    write_json_result(json_result, "urn", "ve", prefix=prefix)
