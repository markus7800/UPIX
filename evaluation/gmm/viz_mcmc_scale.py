
from run_scale import *
import matplotlib.pyplot as plt
import pickle

gt_cluster_visits = jnp.array([687, 574, 119783, 33258676, 46000324, 16768787, 3302321, 485045, 57502, 5806, 457, 38])
gt_ps = gt_cluster_visits / gt_cluster_visits.sum()
gt_cdf = jnp.cumsum(gt_ps)

# for i in range(len(gt_ps)):
#     print(f"{i+1:2d} {gt_ps[i]:.8f}")
#  1 0.00000687
#  2 0.00000574
#  3 0.00119783
#  4 0.33258677
#  5 0.46000323
#  6 0.16768786
#  7 0.03302321
#  8 0.00485045
#  9 0.00057502
# 10 0.00005806
# 11 0.00000457
# 12 0.00000038

n_slps = 8
n_samples_per_chain = 2048
args.n_slps = n_slps

m = gmm(ys)
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_K)

n_chains_to_W1_distance: Dict[int,jax.Array] = dict()
n_chains_to_infty_distance: Dict[int,jax.Array] = dict()

repetitions = 10

with open("viz_gmm_result.txt", "w") as f:
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
            
            W1_distance, infty_distance = get_distance_to_gt(result)

            print(f"W1_distance={W1_distance.item():.8f} infty_distance={infty_distance.item():.8f}")
            f.write(f"n_chains={n_chains} seed={seed} W1_distance={W1_distance.item()} infty_distance={infty_distance.item()}\n")
            f.flush()
                
            W1_distances.append(W1_distance)
            infty_distances.append(infty_distance)
        
        n_chains_to_W1_distance[n_chains] = jnp.vstack(W1_distances).reshape(-1)
        n_chains_to_infty_distance[n_chains] = jnp.vstack(infty_distances).reshape(-1)
    

with open("viz_gmm_mcmc_scale_data.pkl", "wb") as f:
    pickle.dump((n_chains_to_W1_distance, n_chains_to_infty_distance), f)
        
