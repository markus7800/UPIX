
import jax

import jax.numpy as jnp
from tqdm.auto import tqdm
from typing import List
from upix.types import _unstack_sample_data
from typing import Any, Dict
from dataclasses import dataclass
from upix.parallelisation import VectorisationType, parallel_run, parallel_map


from gmm import *
from gibbs_proposals import *
from reversible_jumps import *

class DCCConfig(MCMCDCC[T]):
    def __init__(self, model: Model, return_map: Callable[[Trace], T] = lambda trace: trace, *ignore, verbose=0, **config_kwargs) -> None:
        super().__init__(model, return_map, *ignore, verbose=verbose, **config_kwargs)

    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        return MCMCSteps(
            MCMCStep(SingleVariable("w"), MH(WProposal(delta, slp.decision_representative["K"].item()))),
            MCMCStep(SingleVariable("mus"), MH(MusProposal(ys, kappa, xi, slp.decision_representative["K"].item()))),
            MCMCStep(SingleVariable("vars"), MH(VarsProposal(ys, alpha, beta, slp.decision_representative["K"].item()))),
            MCMCStep(SingleVariable("zs"), MH(ZsProposal(ys))),
        )
    
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
        split_transition_log_prob = estimate.transition_log_probs[(K, K+1)]
        if split_transition_log_prob > jnp.log(0.01):
            trace, _ = self.model.generate(rng_key, {"K": jnp.array(K+1,int)})
            slp = slp_from_decision_representative(self.model, trace)
            active_slps.append(slp)
            tqdm.write(f"Make SLP {slp.formatted()} active.")

    def make_estimate_log_weight_task(self, slp: SLP, rng_key: jax.Array) -> EstimateLogWeightTask:
        last_inference_result = self.inference_results[slp][-1]
        assert isinstance(last_inference_result, MCMCInferenceResult)
        assert not self.config.get("mcmc_optimise_memory_with_early_return_map", False)

        K = find_K(slp)

        def _split(X: Trace, lp: FloatArray, rng_key: PRNGKey):
            return split_move(X, lp, rng_key, K, ys, self.model.log_prob)
        
        def _merge(X: Trace, lp: FloatArray, rng_key: PRNGKey):
            return merge_move(X, lp, rng_key, K, ys, self.model.log_prob)
        
        batch_axis_size = last_inference_result.value_tree[1].size if last_inference_result.value_tree is not None else last_inference_result.last_state.log_prob.size
        split = parallel_map(_split, 0, 0, batch_axis_size, self.pconfig, promote_to_global=True)
        merge = parallel_map(_merge, 0, 0, batch_axis_size, self.pconfig, promote_to_global=True)

        def _task(rng_key: PRNGKey, last_inference_result: MCMCInferenceResult):
            if last_inference_result.value_tree is not None:
                traces: Trace = last_inference_result.value_tree[0]
                lps: FloatArray = last_inference_result.value_tree[1]
                traces = jax.tree.map(_unstack_sample_data, traces)
                lps = _unstack_sample_data(lps)
                assert batch_axis_size == self.mcmc_n_chains * self.mcmc_n_samples_per_chain
            else:
                traces: Trace = last_inference_result.last_state.position
                lps = last_inference_result.last_state.log_prob
                assert batch_axis_size == self.mcmc_n_chains
            n_samples = lps.size
            assert n_samples == batch_axis_size

            transition_log_probs: Dict[Any, FloatArray] = dict()
            
            _split_transition_log_prob = split(traces, lps, jax.random.split(rng_key, lps.shape[0]))
            split_transition_log_prob = jax.scipy.special.logsumexp(_split_transition_log_prob) - jnp.log(n_samples)
            if K == 0: 
                transition_log_probs[(K, K+1)] = split_transition_log_prob
            else:
                _merge_transition_log_prob = merge(traces, lps, jax.random.split(rng_key, lps.shape[0]))
                merge_transition_log_prob = jax.scipy.special.logsumexp(_merge_transition_log_prob) - jnp.log(n_samples)

                merge_transition_log_prob = merge_transition_log_prob - jnp.log(2)
                split_transition_log_prob = split_transition_log_prob - jnp.log(2)

                transition_log_probs[(K, K+1)] = split_transition_log_prob
                transition_log_probs[(K, K-1)] = merge_transition_log_prob

            return RJMCMCTransitionProbEstimate(transition_log_probs, n_samples)
        
        def _post_info(result: LogWeightEstimate):
            assert isinstance(result, RJMCMCTransitionProbEstimate)
            if K == 0:
                split_transition_log_prob = result.transition_log_probs[(K, K+1)]
                return f"Estimate log weight for {slp.formatted()}: split prob = {jnp.exp(split_transition_log_prob.item()):.6f}"
            else:
                split_transition_log_prob = result.transition_log_probs[(K, K+1)]
                merge_transition_log_prob = result.transition_log_probs[(K, K-1)]
                return f"Estimate log weight for {slp.formatted()}: split prob = {jnp.exp(split_transition_log_prob).item():.6f} merge prob = {jnp.exp(merge_transition_log_prob).item():.6f}"
        
        return EstimateLogWeightTask(_task, (rng_key,last_inference_result), post_info=_post_info)

    
    def compute_slp_log_weight(self, log_weight_estimates: Dict[SLP, LogWeightEstimate]) -> Dict[SLP, FloatArray]:
        max_K = max(find_K(slp) for slp in log_weight_estimates.keys())
        visited_k = jnp.zeros((max_K+1,))
        bayes_factor = jnp.zeros((max_K+1, max_K+1))
        for estimate in log_weight_estimates.values():
            assert isinstance(estimate, RJMCMCTransitionProbEstimate)
            for (current_k, next_k), log_prob in estimate.transition_log_probs.items():
                visited_k = visited_k.at[current_k].set(1)
                if next_k <= max_K:
                    bayes_factor = bayes_factor.at[current_k, next_k].add(log_prob)
                    bayes_factor = bayes_factor.at[next_k, current_k].add(-log_prob)
        assert jnp.all(visited_k == 1)
        for i in range(max_K+1):
            for j in range(i+1,max_K+1):
                bayes_factor = bayes_factor.at[i, j].set(bayes_factor[i,j-1] - bayes_factor[j, j-1])
                bayes_factor = bayes_factor.at[j, i].set(-bayes_factor[i,j])
        
        k_to_log_weight = -jax.scipy.special.logsumexp(bayes_factor, axis=1)

        result: Dict[SLP, FloatArray] = dict()
        for slp in log_weight_estimates.keys():
            k = find_K(slp)
            result[slp] = k_to_log_weight[k]

        return result
