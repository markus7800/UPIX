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

from dccxjax.infer.mcmc.metropolis import MHInfo

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
    
    def get_initial_positions(self, slp: SLP, rng_key: PRNGKey) -> StackedTrace:
        k = slp.decision_representative["K"].item()
        @jax.jit
        @jax.vmap
        def get_positions(key: PRNGKey):
            trace, _ = slp.generate(key, {"K": jnp.array(k,int)})
            return trace
        traces = get_positions(jax.random.split(rng_key, self.mcmc_n_chains))
        
        # check if all initial positions are in SLP support
        # _, _, pcs = jax.vmap(slp._log_prior_likeli_pathcond, in_axes=(0,None))(traces, dict())
        # tqdm.write(f"Mean pc {jnp.mean(pcs)}.")
        
        return StackedTrace(traces, self.mcmc_n_chains)


if __name__ == "__main__":
    m = gmm(ys)
    m.set_slp_formatter(formatter)
    m.set_slp_sort_key(find_K)
    print(f"N_CHAINS={args.n_chains} N_SAMPLES_PER_CHAIN={args.n_samples_per_chain}")

    dcc_obj = StaticDCCConfig(m, verbose=2,
        mcmc_n_chains=args.n_chains,
        mcmc_n_samples_per_chain=args.n_samples_per_chain,
        mcmc_collect_for_all_traces=False,
        parallelisation=get_parallelisation_config(args),
        disable_progress=args.no_progress
    )

    result, timings = timed(dcc_obj.run)(jax.random.key(0))
    result.pprint(sortkey="slp")
    
    workload = {
        "n_chains": args.n_chains,
        "n_samples_per_chain": args.n_samples_per_chain,
        "n_slps": len(result.get_slps())
    }
    
    avg_acceptance_rate = jnp.mean(jnp.vstack(
        [jnp.vstack([info.accepted/args.n_samples_per_chain for info in r[0].last_state.infos if isinstance(info, MHInfo)])
         for r in dcc_obj.inference_results.values() if isinstance(r[0], MCMCInferenceResult) and r[0].last_state.infos is not None]))

    result_metrics = {
        "result_str": result.sprint(sortkey="slp"),
        "avg_acceptance_rate": avg_acceptance_rate.item(),
        "pmap_check": str(check_pmap())
    }
        
    json_result = {
        "workload": workload,
        "timings": timings,
        "dcc_timings": dcc_obj.get_timings(),
        "result_metrics": result_metrics,
        "args": args.__dict__,
        "pconfig": dcc_obj.pconfig.__dict__,
        "environment_info": get_environment_info()
    }
    
    if not args.no_save:
        prefix = f"nchains_{args.n_chains:07d}_nslps_{len(result.get_slps())}_niter_{args.n_samples_per_chain}_"
        write_json_result(json_result, "gmm", "scale", prefix=prefix)
