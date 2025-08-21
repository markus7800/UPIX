import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("--show_plots", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)

from dccxjax.core import *
from dccxjax.viz import *
from setup_parallelisation import get_parallelisation_config

from gp_smc import *

AutoGPConfig()

if __name__ == "__main__":
    
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    m.set_equivalence_map(equivalence_map)

    smc_dcc_obj = SMCDCCConfig(m, verbose=2,
        smc_rejuvination_attempts=8,
        smc_n_particles=100,
        smc_collect_inference_info=True,
        max_iterations = 5,
        n_lmh_update_samples = 250,
        max_active_slps = 3,
        max_new_active_slps = 3,
        one_inference_run_per_slp = True,
        parallelisation = get_parallelisation_config(args)
    )

    result = timed(smc_dcc_obj.run)(jax.random.key(0))
    result.pprint()

    if args.show_plots:
        slp_weights = list(result.get_slp_weights().items())
        slp_weights.sort(key=lambda v: v[1])

        map_slp, _ = slp_weights[-1]

        weighted_samples = result.get_samples_for_slp(map_slp).unstack()
        _, weights = weighted_samples.get()

        map_trace, _ = weighted_samples.get_selection(weights.argmax())
        k = get_gp_kernel(map_trace)
        noise = transform_param("noise", map_trace["noise"]) + 1e-5
        xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))
        mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)

        plt.figure()
        plt.scatter(xs, ys)
        plt.scatter(xs_val, ys_val)
        plt.plot(xs_pred, mvn.numpyro_base.mean, color="black")
        q025, q975 = mvnormal_quantile(mvn, 0.025), mvnormal_quantile(mvn, 0.975)
        plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
        plt.title(map_slp.formatted())
        plt.show()



        plt.figure()
        plt.scatter(xs, ys)
        plt.scatter(xs_val, ys_val)
        xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))

        n_posterior_samples = 1_000
        sample_key = jax.random.key(0)
        slp_weights_array = jnp.array([weight for _, weight in slp_weights])
        posterior_over_slps = dist.Categorical(slp_weights_array)

        samples = []
        for i in tqdm(range(n_posterior_samples), desc="Sample posterior"):
            sample_key, key1, key2, key3 = jax.random.split(sample_key, 4)
            slp_ix = posterior_over_slps.sample(key1)
            slp, _ = slp_weights[slp_ix]
            weighted_samples = result.get_samples_for_slp(slp).unstack()
            _, weights = weighted_samples.get()
            trace_ix = dist.Categorical(weights).sample(key2)
            trace, weight = weighted_samples.get_selection(trace_ix)
            
            k = get_gp_kernel(trace)
            noise = transform_param("noise", trace["noise"]) + 1e-5
            mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)


            if i < 10:
                tqdm.write(f"sample from posterior: {slp.formatted()} noise={noise} with log_prob {m.log_prob(trace)}")

                plt.plot(xs_pred, mvn.numpyro_base.mean, color="black", alpha=0.1)
                q025, q975 = mvnormal_quantile(mvn, 0.025), mvnormal_quantile(mvn, 0.975)
                plt.fill_between(xs_pred, q025, q975, alpha=0.1, color="tab:blue")

            samples.append(mvn.sample(key3))
        # plt.show()


        samples = jnp.vstack(samples)
        q050 = jnp.median(samples, axis=0)
        q025 = jnp.quantile(samples, 0.025, axis=0)
        q975 = jnp.quantile(samples, 0.975, axis=0)

        plt.figure()
        plt.scatter(xs, ys)
        plt.scatter(xs_val, ys_val)
        plt.plot(xs_pred, q050, color="black")
        plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
        plt.show()
