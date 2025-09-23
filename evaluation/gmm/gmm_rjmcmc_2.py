
import jax

import jax.numpy as jnp
from tqdm.auto import tqdm
from typing import List, Callable
from dccxjax.types import _unstack_sample_data
from typing import Any, Dict
from dataclasses import dataclass
from dccxjax.parallelisation import VectorisationType, parallel_run, parallel_map
from dccxjax.core.branching_tracer import trace_branching, retrace_branching


from gmm import *
from gibbs_proposals import *
from reversible_jumps_2 import *

class DCCConfig(RJMCMCDCC[T]):
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        return MCMCSteps(
            MCMCStep(SingleVariable("w"), MH(WProposal(delta, slp.decision_representative["K"].item()))),
            MCMCStep(SingleVariable("mus"), MH(MusProposal(ys, kappa, xi, slp.decision_representative["K"].item()))),
            MCMCStep(SingleVariable("vars"), MH(VarsProposal(ys, alpha, beta, slp.decision_representative["K"].item()))),
            MCMCStep(SingleVariable("zs"), MH(ZsProposal(ys))),
        )
    
    def get_index(self, trace: Trace) -> IntArray:
        return trace["K"]
    
    def get_reversible_jumps(self, slp: SLP) -> List[ReversibleJump]:
        K = find_K(slp)
        jumps = [ReversibleJump(get_split_move(K), get_split_aux(K, ys), get_merge_move(K+1), get_merge_aux(K+1))]
        if K > 0:
            jumps.append(ReversibleJump(get_merge_move(K), get_merge_aux(K), get_split_move(K-1), get_split_aux(K-1, ys)))
        return jumps
    
    
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        rng_key, generate_key = jax.random.split(rng_key)
        trace, _ = self.model.generate(generate_key, {"K": jnp.array(0,int)})
        slp = slp_from_decision_representative(self.model, trace)
        active_slps.append(slp)
        tqdm.write(f"Make SLP {slp.formatted()} active.")

    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        last_slp = active_slps.pop()
        inactive_slps.append(last_slp)
        estimates = self.get_log_weight_estimates(last_slp)
        estimate = estimates[0] # we only have one per slp
        assert isinstance(estimate, RJMCMCTransitionProbEstimate)
        K = find_K(last_slp)
        split_transition_prob = estimate.transition_probs[K+1]
        if split_transition_prob > 0.01:
            trace, _ = self.model.generate(rng_key, {"K": jnp.array(K+1,int)})
            slp = slp_from_decision_representative(self.model, trace)
            active_slps.append(slp)
            tqdm.write(f"Make SLP {slp.formatted()} active.")
