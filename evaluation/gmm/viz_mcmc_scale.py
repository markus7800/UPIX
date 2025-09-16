
from run_scale import *
import matplotlib.pyplot as plt
import pickle

gt_cluster_visits = jnp.array([687, 574, 119783, 33258676, 46000324, 16768787, 3302321, 485045, 57502, 5806, 457, 38])
gt_ps = gt_cluster_visits / gt_cluster_visits.sum()
gt_cdf = jnp.cumsum(gt_ps)

n_slps = 10
n_samples_per_chain = 2024
args.n_slps = n_slps

m = gmm(ys)
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_K)

n_chains_to_W1_distance: Dict[str,jax.Array] = dict()
n_chains_to_infty_distance: Dict[str,jax.Array] = dict()

repetitions = 10

for n_chains in [2**n for n in range(18+1)]:
    W1_distances = []
    infty_distances = []
    for seed in range(repetitions):
        dcc_obj = StaticDCCConfig(m, verbose=2,
            mcmc_n_chains=n_chains,
            mcmc_n_samples_per_chain=n_samples_per_chain,
            mcmc_collect_for_all_traces=False,
            parallelisation=get_parallelisation_config(args),
            disable_progress=args.no_progress
        )
        result = dcc_obj.run(jax.random.key(seed))
        result.pprint(sortkey="slp")
        lps = jnp.array([log_weight for _, log_weight in result.get_log_weights_sorted("slp")])
        lps = lps - jax.scipy.special.logsumexp(lps)
        ps = jax.lax.exp(lps)
        ps = ps / ps.sum()
        cdf_est = jnp.cumsum(ps)
        print(ps, gt_ps)
        
        W1_distance = jnp.trapezoid(jnp.abs(cdf_est - gt_cdf[:cdf_est.size]))
        infty_distance = jnp.max(jnp.abs(cdf_est - gt_cdf[:cdf_est.size]))
        
        W1_distances.append(W1_distance)
        infty_distances.append(infty_distance)
    
    n_chains_to_W1_distance[f"{n_chains:,}"] = jnp.vstack(W1_distances).reshape(-1)
    n_chains_to_infty_distance[f"{n_chains:,}"] = jnp.vstack(infty_distances).reshape(-1)
    

with open("viz_gmm_mcmc_scale_data.pkl", "wb") as f:
    pickle.dump((n_chains_to_W1_distance, n_chains_to_infty_distance), f)
        
        

fig, ax = plt.subplots()
plt.title("W1 distance")
ax.boxplot(n_chains_to_W1_distance.values()) # type: ignore
ax.set_xticklabels(n_chains_to_W1_distance.keys())

fig, ax = plt.subplots()
plt.title("Infty distance")
ax.boxplot(n_chains_to_infty_distance.values()) # type: ignore
ax.set_xticklabels(n_chains_to_infty_distance.keys())
plt.show()
    

