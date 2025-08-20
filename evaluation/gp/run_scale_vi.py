import sys
from typing import List

sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("n_slps", help="number of slps to evaluate", type=int)
parser.add_argument("L", help="number of samples to take per ADVI iteration", type=int)
parser.add_argument("n_iter", help="number of ADVI iterations", type=int)
parser.add_argument("--show_plots", action="store_true")
args = parser.parse_args()
setup_devices_from_args(args)

from dccxjax.core import *
from dccxjax.viz import *
from setup_parallelisation import get_parallelisation_config

from gp_vi import *
from dccxjax.infer.variational_inference.optimizers import Adagrad, SGD, Adam

from enumerate_slps import find_active_slps_through_enumeration

# AutoGP
# N_LEAF_NODE_TYPES = 5
# NODE_TYPES: List[type[GPKernel]] = [Constant, Linear, SquaredExponential, GammaExponential, Periodic, Plus, Times]
# NODE_TYPE_PROBS = normalise(jnp.array([0, 6, 0, 6, 6, 5, 5],float))

# Reichelt
N_LEAF_NODE_TYPES = 4
NODE_TYPES: List[type[GPKernel]] = [UnitRationalQuadratic, UnitPolynomialDegreeOne, UnitSquaredExponential, UnitPeriodic, Plus, Times]
NODE_TYPE_PROBS = normalise(jnp.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1],float))

class VIConfig2(VIConfig):
    def __init__(self, model: Model, *ignore, verbose=0, **config_kwargs) -> None:
        VIDCC.__init__(self, model, *ignore, verbose=verbose, **config_kwargs)

    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        find_active_slps_through_enumeration(N_LEAF_NODE_TYPES, active_slps, rng_key, args.n_slps, self.model)
    
    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        inactive_slps.clear()
        inactive_slps.extend(active_slps)
        active_slps.clear()

if __name__ == "__main__":
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    m.set_equivalence_map(equivalence_map)
    
    vi_dcc_obj = VIConfig2(m, verbose=2,
        advi_n_iter = args.n_iter,
        advi_L=args.L,
        advi_optimizer=Adam(0.005),
        elbo_estimate_n_samples=100,
        parallelisation = get_parallelisation_config(args)
    )

    result = timed(vi_dcc_obj.run)(jax.random.key(0))
    result.pprint()

    if args.show_plots:
        slp_weights = list(result.get_slp_weights().items())
        slp_weights.sort(key=lambda v: v[1])

        xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))
        slp, weight = slp_weights[-1]
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
    
