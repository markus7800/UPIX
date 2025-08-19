import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
args = parser.parse_args()
setup_devices_from_args(args)

from dccxjax.core import *
from setup_parallelisation import get_parallelisation_config

import logging
setup_logging(logging.WARNING)

from gmm import *

if __name__ == "__main__":
    m = gmm(ys)
    m.set_slp_formatter(formatter)
    m.set_slp_sort_key(find_K)

    dcc_obj = DCCConfig(m, verbose=2,
        mcmc_n_chains=16,
        mcmc_n_samples_per_chain=25_000,
        mcmc_collect_for_all_traces=True,
        parallelisation=get_parallelisation_config(args)
    )

    # takes ~185s for 10 * 25_000 * 11 samples
    result = timed(dcc_obj.run)(jax.random.key(0))
    result.pprint(sortkey="slp")