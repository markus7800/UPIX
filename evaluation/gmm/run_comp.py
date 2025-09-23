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

from gmm_rjmcmc_2 import *

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

    result, timings = timed(dcc_obj.run)(jax.random.key(0))
    result.pprint(sortkey="slp")
    
    
    gt_cluster_visits = jnp.array([687, 574, 119783, 33258676, 46000324, 16768787, 3302321, 485045, 57502, 5806, 457, 38])
    gt_ps = gt_cluster_visits / gt_cluster_visits.sum()
    gt_cdf = jnp.cumsum(gt_ps)
    
    lps = jnp.array([log_weight for _, log_weight in result.get_log_weights_sorted("slp")])
    lps = lps - jax.scipy.special.logsumexp(lps)
    ps = jax.lax.exp(lps)
    ps = ps / ps.sum()
    ps = jax.lax.concatenate((ps, jnp.zeros((len(gt_ps)-len(ps),))), 0)
    cdf_est = jnp.cumsum(ps)
    for i in range(len(ps)):
        print(f"{i:2d}: {ps[i]:.8f} - {gt_ps[i]:.8f} = {ps[i] - gt_ps[i]:.8f}")

    W1_distance = jnp.trapezoid(jnp.abs(cdf_est - gt_cdf))
    infty_distance = jnp.max(jnp.abs(cdf_est - gt_cdf))
    
    print(f"W1_distance={W1_distance.item():.8f} infty_distance={infty_distance.item():.8f}")