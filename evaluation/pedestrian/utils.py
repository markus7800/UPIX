
from upix.core import *
from upix.viz import *
from upix.infer.mcmc.hmc import HMCInfo
from pedestrian import *

from upix.infer import MCMCDCC, T, MCMCInferenceResult
from upix.infer.dcc.mc_dcc import MCDCCResult

def save_results(args, result: MCDCCResult[Trace], dcc_obj: MCMCDCC[T], timings: dict, W1_distance: float, infty_distance : float, folder: str):
    workload = {
        "n_chains": dcc_obj.mcmc_n_chains,
        "n_samples_per_chain": dcc_obj.mcmc_n_samples_per_chain,
        "n_slps": len(result.get_slps()),
        "seed": args.seed
    }
    
    avg_acceptance_rate = jnp.mean(jnp.vstack(
        [jnp.vstack([info.accepted/dcc_obj.mcmc_n_samples_per_chain for info in r[0].last_state.infos if isinstance(info, HMCInfo)])
         for r in dcc_obj.inference_results.values() if isinstance(r[0], MCMCInferenceResult) and r[0].last_state.infos is not None]))

    result_metrics = {
        "W1": W1_distance,
        "L_inf": infty_distance,
        "result_str": result.sprint(sortkey="slp"),
        "avg_acceptance_rate": avg_acceptance_rate.item(),
    }
        
    json_result = {
        "workload": workload,
        "timings": timings,
        "dcc_timings": dcc_obj.get_timings(),
        "result_metrics": result_metrics,
        "args": args.__dict__,
        "pconfig": dcc_obj.pconfig.__dict__,
        "environment_info":  get_environment_info()
    }
    
    prefix = f"nchains_{dcc_obj.mcmc_n_chains:07d}_nslps_{len(result.get_slps())}_niter_{dcc_obj.mcmc_n_samples_per_chain}_"
    write_json_result(json_result, "pedestrian", folder, prefix=prefix)

