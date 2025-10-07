import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("--show_plots", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)

from upix.core import *
from setup_parallelisation import get_parallelisation_config
import matplotlib.pyplot as plt

import logging
setup_logging(logging.WARNING)

from gmm_rjmcmc import *

if __name__ == "__main__":
    m = gmm(ys)
    m.set_slp_formatter(formatter)
    m.set_slp_sort_key(find_K)

    dcc_obj = DCCConfig(m, verbose=2,
        mcmc_n_chains=8,
        mcmc_n_samples_per_chain=25_000,
        mcmc_collect_for_all_traces=True,
        parallelisation=get_parallelisation_config(args)
    )

    result, timings = timed(dcc_obj.run)(jax.random.key(0))
    result.pprint(sortkey="slp")
    
    W1_distance, infty_distance = get_distance_to_gt(result)
    
    print(f"W1_distance={W1_distance.item():.8f} infty_distance={infty_distance.item():.8f}")
    
    if args.show_plots:
        slps: List[SLP] = []
        log_weights = []
        for slp, log_weight in result.slp_log_weights.items():
            slps.append(slp)
            log_weights.append(log_weight)
        log_weights = jnp.array(log_weights)
        
        fig = plt.figure()
        y_space = jnp.linspace(ys.min()-1, ys.max()+1, 512)
        
        colors= ["black", "black", "black", "tab:blue", "tab:orange", "tab:green", "tab:red", "black", "black", "black"]
        keys = jax.random.split(jax.random.key(0), 256)
        for key in keys:
            slp_key, trace_key = jax.random.split(key)
            slp_ix = jax.random.categorical(slp_key, log_weights)
            trace_ix = jax.random.randint(trace_key, (), 0, dcc_obj.mcmc_n_chains * dcc_obj.mcmc_n_samples_per_chain)
            slp = slps[slp_ix]
            slps_samples = result.get_samples_for_slp(slp)
            trace, _ = slps_samples.unstack().get_ix(int(trace_ix))
            
            w = trace["w"].reshape(-1,1)
            lps = dist.Normal(trace["mus"].reshape(-1,1), trace["vars"].reshape(-1,1)).log_prob(y_space.reshape(1,-1))
            p = jnp.sum(w * jnp.exp(lps), axis=0)
            plt.plot(y_space, p, color="tab:blue", alpha=0.05)
            # for k in range(trace["K"]):
            #     plt.plot(y_space, jnp.exp(lps[k,:]), color=colors[trace["K"]], alpha=0.1)
            
        plt.scatter(ys, jnp.full_like(ys,0), marker="x", color="black")
        plt.show()
        