import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("--show_plots", action="store_true")
parser.add_argument("n_slps", type=int)
args = parser.parse_args()
setup_devices_from_args(args)

from dccxjax.core import *
from dccxjax.viz import *
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
        jit_inference=True,
        N_max = args.n_slps
    )

    result = timed(config.run)(jax.random.key(0))
    result.pprint(sortkey="slp")

    gt = jnp.load("evaluation/urn/gt_ps.npy")
    log_Zs = jnp.array([log_Z for (slp, log_Z) in result.get_log_weights_sorted(sortkey="slp")])

    assert (jnp.array([slp.decision_representative["N"] for (slp, log_Z) in result.get_log_weights_sorted(sortkey="slp")]) == jnp.arange(1,len(log_Zs)+1)).all()

    ps = jnp.exp(log_Zs - jax.scipy.special.logsumexp(log_Zs))
    print(ps)
    print("vs")
    print(gt[:len(ps)])

    if args.show_plots:
        err = jnp.abs(ps - gt[:len(ps)])
        plt.plot(err)
        plt.show()

        print("Max err: ", jnp.max(err))
