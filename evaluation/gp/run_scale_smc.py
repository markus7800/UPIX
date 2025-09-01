import sys
from typing import List

sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("n_slps", help="number of slps to evaluate", type=int)
parser.add_argument("n_particles", help="number of smc particles", type=int)
parser.add_argument("--show_plots", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)

from dccxjax.core import *
from dccxjax.viz import *
from setup_parallelisation import get_parallelisation_config
from dccxjax.infer import InferenceResult, LogWeightEstimate

from gp_smc import *

from enumerate_slps import find_active_slps_through_enumeration

AutoGPConfig()

class SMCDCCConfig2(SMCDCCConfig[T]):
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        find_active_slps_through_enumeration(NODE_CONFIG.N_LEAF_NODE_TYPES, active_slps, rng_key, args.n_slps, self.model)
    
    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        inactive_slps.clear()
        inactive_slps.extend(active_slps)
        active_slps.clear()

if __name__ == "__main__":
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    m.set_equivalence_map(equivalence_map)
    
    smc_dcc_obj = SMCDCCConfig2(m, verbose=2,
        smc_collect_inference_info=True,
        parallelisation = get_parallelisation_config(args),
        smc_n_partilces = args.n_particles,
        disable_progress=args.no_progress
    )

    result = timed(smc_dcc_obj.run)(jax.random.key(0))
    result.pprint()

    if args.show_plots:
        slp_weights = list(result.get_slp_weights().items())
        slp_weights.sort(key=lambda v: v[1])

        xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))
        
        for slp_ix in range(1,5+1):
            slp, weight = slp_weights[-slp_ix]
            print(slp.formatted(), weight)
            
            weighted_samples = result.get_samples_for_slp(slp).unstack()
            _, weights = weighted_samples.get()

            xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))

            n_posterior_samples = 1_000

            samples = []
            sample_key = jax.random.key(0)
            for i in tqdm(range(n_posterior_samples), desc="Sample posterior of MAP SLP"):
                sample_key, key1, key2 = jax.random.split(sample_key, 3)

                trace_ix = dist.Categorical(weights).sample(key1)
                trace, weight = weighted_samples.get_selection(trace_ix)
                
                k = get_gp_kernel(trace)
                noise = transform_param("noise", trace["noise"]) + 1e-5
                mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)

                samples.append(mvn.sample(key2))

            samples = jnp.vstack(samples)
            m = jnp.mean(samples, axis=0)
            q025 = jnp.quantile(samples, 0.025, axis=0)
            q975 = jnp.quantile(samples, 0.975, axis=0)

            plt.figure()
            plt.title(f"{slp_ix}. {slp.formatted()}")
            plt.scatter(xs, ys)
            plt.scatter(xs_val, ys_val)
            plt.plot(xs_pred, m, color="black")
            plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
        plt.show()