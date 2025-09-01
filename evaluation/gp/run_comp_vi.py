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

from gp_vi import *
from dccxjax.infer.variational_inference.optimizers import Adagrad, SGD, Adam

RecheiltConfig()

if __name__ == "__main__":
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    m.set_equivalence_map(equivalence_map)
    
    vi_dcc_obj = VIConfig(m, verbose=2,
        init_n_samples=1_000,
        advi_L=1,
        advi_optimizer=Adam(0.005),
        elbo_estimate_n_samples=100,
        successive_halving=SuccessiveHalving(1_000_000, 10),
        parallelisation = get_parallelisation_config(args),
        disable_progress=args.no_progress
    )

    result = timed(vi_dcc_obj.run)(jax.random.key(0))
    result.pprint()

    if args.show_plots:
        slp_weights = list(result.get_slp_weights().items())
        slp_weights.sort(key=lambda v: v[1])

        xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))
        for i in range(min(len(slp_weights),5)):
            slp, weight = slp_weights[-(i+1)]
            print(slp.formatted(), weight)
            g = result.slp_guides[slp]
            
            n = 100
            
            key = jax.random.key(0)
            posterior = Traces(g.sample(key, (n,)), n)
            
            samples = []
            for i in range(n):
                key, sample_key = jax.random.split(key)
                trace = posterior.get_ix(i)
                k = get_gp_kernel(trace)
                noise = transform_param("noise", trace["noise"]) + 1e-5
                mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)
                samples.append(mvn.sample(sample_key))

            samples = jnp.vstack(samples)
            m = jnp.mean(samples, axis=0)
            q025 = jnp.quantile(samples, 0.025, axis=0)
            q975 = jnp.quantile(samples, 0.975, axis=0)

            plt.figure()
            plt.title(slp.formatted())
            plt.scatter(xs, ys)
            plt.scatter(xs_val, ys_val)
            plt.plot(xs_pred, m, color="black")
            plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
        plt.show()
    
