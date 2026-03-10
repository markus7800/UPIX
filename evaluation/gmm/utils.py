
from upix.core import *

import logging
setup_logging(logging.WARNING)

from gmm_rjmcmc import *
from upix.infer.mcmc.metropolis import MHInfo


def save_result(args, result: MCDCCResult[Trace], dcc_obj: DCCConfig, timings: dict, W1_distance: float, infty_distance: float, folder: str):
    workload = {
        "n_chains": dcc_obj.mcmc_n_chains,
        "n_samples_per_chain": dcc_obj.mcmc_n_samples_per_chain,
        "n_slps": len(result.get_slps()),
        "seed": args.seed
    }
    
    avg_acceptance_rate = jnp.mean(jnp.vstack(
        [jnp.vstack([info.accepted/dcc_obj.mcmc_n_samples_per_chain for info in r[0].last_state.infos if isinstance(info, MHInfo)])
         for r in dcc_obj.inference_results.values() if isinstance(r[0], MCMCInferenceResult) and r[0].last_state.infos is not None]))

    result_metrics = {
        "result_str": result.sprint(sortkey="slp"),
        "avg_acceptance_rate": avg_acceptance_rate.item(),
        "W1": W1_distance,
        "L_inf": infty_distance
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
    
    prefix = f"nchains_{dcc_obj.mcmc_n_chains:07d}_nslps_{len(result.get_slps())}_niter_{dcc_obj.mcmc_n_samples_per_chain}_"
    write_json_result(json_result, "gmm", folder, prefix=prefix)