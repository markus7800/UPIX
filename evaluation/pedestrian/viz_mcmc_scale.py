
from run_scale import *
import matplotlib.pyplot as plt
import pickle

gt_xs = jnp.load("evaluation/pedestrian/gt_xs-100.npy")
gt_cdf = jnp.load("evaluation/pedestrian/gt_cdf-100-1_000_000_000_000.npy")
gt_pdf = jnp.load("evaluation/pedestrian/gt_pdf_est-100-1_000_000_000_000.npy")


n_slps = 10
n_samples_per_chain = 256
args.n_slps = n_slps

m = pedestrian()
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_t_max)


@jax.jit
def cdf_estimate(sample_points, sample_weights: jax.Array, qs):
    def _cdf_estimate(q):
        return jnp.where(sample_points < q, sample_weights, jax.numpy.zeros_like(sample_weights)).sum()
    return jax.lax.map(_cdf_estimate, qs)
        
n_chains_to_W1_distance: Dict[int,jax.Array] = dict()
n_chains_to_infty_distance: Dict[int,jax.Array] = dict()

repetitions = 10

for n_chains in [2**n for n in range(20+1)]:
    W1_distances = []
    infty_distances = []
    for seed in range(repetitions):
        dcc_obj = DCCConfig(m, verbose=2,
                parallelisation=get_parallelisation_config(args),
                init_n_samples=250,
                init_estimate_weight_n_samples=2**20, # ~10**6
                mcmc_n_chains=n_chains,
                mcmc_n_samples_per_chain=n_samples_per_chain,
                estimate_weight_n_samples=2**23, # ~10**7
                max_iterations=1,
                mcmc_collect_for_all_traces=False,
                mcmc_optimise_memory_with_early_return_map=True,
                return_map=lambda trace: {"start": trace["start"]},
                disable_progress=args.no_progress
        )
        result = dcc_obj.run(jax.random.key(seed))
        result.pprint(sortkey="slp")
        
        start_weighted_samples, _ = result.get_samples_for_address("start") 
        assert start_weighted_samples is not None
        start_samples, start_weights = start_weighted_samples.get()

        cdf_est = cdf_estimate(start_samples, start_weights, gt_xs)
        W1_distance = jnp.trapezoid(jnp.abs(cdf_est - gt_cdf), gt_xs)
        infty_distance = jnp.max(jnp.abs(cdf_est - gt_cdf))
        
        W1_distances.append(W1_distance)
        infty_distances.append(infty_distance)
    
    n_chains_to_W1_distance[n_chains] = jnp.vstack(W1_distances).reshape(-1)
    n_chains_to_infty_distance[n_chains] = jnp.vstack(infty_distances).reshape(-1)
    
with open("viz_ped_mcmc_scale_data.pkl", "wb") as f:
    pickle.dump((n_chains_to_W1_distance, n_chains_to_infty_distance), f)
    