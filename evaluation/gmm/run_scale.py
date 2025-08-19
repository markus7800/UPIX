import sys
sys.path.append("evaluation")
from parse_args import *
parser = get_arg_parser()
parser.add_argument("n_slps", help="number of slps to evaluate", type=int)
parser.add_argument("n_chains", help="number of chains to run", type=int)
parser.add_argument("n_samples_per_chain", help="number of sampler per chain to run", type=int)
args = parser.parse_args()
setup_devices_from_args(args)

from dccxjax.core import *
from setup_parallelisation import get_parallelisation_config

import logging
setup_logging(logging.WARNING)

from gmm import *



class StaticDCCConfig(DCCConfig):
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        for i in range(args.n_slps):
            rng_key, generate_key = jax.random.split(rng_key)
            trace, _ = self.model.generate(generate_key, {"K": jnp.array(i,int)})
            slp = slp_from_decision_representative(self.model, trace)
            active_slps.append(slp)
            tqdm.write(f"Make SLP {slp.formatted()} active.")

    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        inactive_slps.extend(active_slps)
        active_slps.clear()


m = gmm(ys)
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_K)

dcc_obj = DCCConfig(m, verbose=2,
    mcmc_n_chains=args.n_chains,
    mcmc_n_samples_per_chain=args.n_samples_per_chain,
    mcmc_collect_for_all_traces=False,
    parallelisation=get_parallelisation_config(args)
)

# takes ~185s for 10 * 25_000 * 11 samples
result = timed(dcc_obj.run)(jax.random.key(0))
result.pprint(sortkey="slp")