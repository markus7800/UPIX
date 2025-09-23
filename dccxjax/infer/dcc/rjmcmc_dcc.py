
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
from typing import List, Callable
from dccxjax.types import _unstack_sample_data
from typing import Any, Dict
from dataclasses import dataclass
from dccxjax.parallelisation import VectorisationType, parallel_run, parallel_map
from dccxjax.core.branching_tracer import trace_branching, retrace_branching
from dccxjax.core import Model, SLP
from dccxjax.types import FloatArray, Trace, IntArray, PRNGKey
from dccxjax.infer.dcc import LogWeightEstimate, MCMCDCC, DCC_COLLECT_TYPE, EstimateLogWeightTask, MCMCInferenceResult
from abc import ABC, abstractmethod
from dccxjax.infer.involutive import TraceTansformation
from functools import partial


__all__ = [
    "ReversibleJump",
    "RJMCMCDCC",
    "RJMCMCTransitionProbEstimate",
]

@jax.tree_util.register_dataclass
@dataclass
class RJMCMCTransitionProbEstimate(LogWeightEstimate):
    transition_probs: Dict[int, FloatArray]
    n_samples: int
    def combine_estimates(self, other: LogWeightEstimate) -> "RJMCMCTransitionProbEstimate":
        assert isinstance(other, RJMCMCTransitionProbEstimate)

        n_combined_samples = self.n_samples + other.n_samples
        a = self.n_samples / n_combined_samples
        new_transition_probs: Dict[int, FloatArray] = dict()

        for key in self.transition_probs.keys() | other.transition_probs.keys():
            est1 = self.transition_probs.get(key, jnp.array(0.,float))
            est2 = other.transition_probs.get(key, jnp.array(0.,float))
            new_transition_probs[key] = a * est1 + (1-a) * est2
        return RJMCMCTransitionProbEstimate(new_transition_probs, n_combined_samples)

@dataclass
class ReversibleJump:
    move: Callable[[Trace,Trace,Trace,Trace],None]
    aux_model_gen: Callable[[Trace],Model]
    reverse_move: Callable[[Trace,Trace,Trace,Trace],None] 
    reverse_aux_model_gen: Callable[[Trace],Model]

class RJMCMCDCC(MCMCDCC[DCC_COLLECT_TYPE]):
    
    @abstractmethod
    def get_index(self, trace: Trace) -> IntArray:
        raise NotImplementedError
    
    @abstractmethod
    def get_reversible_jumps(self, slp: SLP) -> List[ReversibleJump]:
        raise NotImplementedError
    
    def make_estimate_log_weight_task(self, slp: SLP, rng_key: jax.Array) -> EstimateLogWeightTask:
        last_inference_result = self.inference_results[slp][-1]
        assert isinstance(last_inference_result, MCMCInferenceResult)
        assert not self.config.get("mcmc_optimise_memory_with_early_return_map", False)

        batch_axis_size = last_inference_result.value_tree[1].size if last_inference_result.value_tree is not None else last_inference_result.last_state.log_prob.size

        def _involutive_kernel(model_trace: Trace, lp: FloatArray, key: PRNGKey, jump: ReversibleJump):
            generate_key, accept_key = jax.random.split(key)
            aux_trace, aux_lp_forward = jump.aux_model_gen(model_trace).generate(generate_key)
            tt = TraceTansformation(jump.move)
            with tt:
                new_model_trace, new_aux_trace = tt.apply(model_trace, aux_trace)
                j = tt.jacobian(model_trace, aux_trace)
            logabsdetJ = jnp.linalg.slogdet(j).logabsdet
            aux_lp_backward = jump.reverse_aux_model_gen(new_model_trace).log_prob(new_aux_trace)
            
            log_alpha = self.model.log_prob(new_model_trace) - lp + aux_lp_backward - aux_lp_forward + logabsdetJ
            accept = jax.lax.log(jax.random.uniform(accept_key)) < log_alpha
            return jax.lax.select(accept, self.get_index(new_model_trace), self.get_index(model_trace))
            
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
        
            slps_ixs = []
            for jump in self.get_reversible_jumps(slp):
                involutive_kernel = parallel_map(partial(_involutive_kernel,jump=jump), 0, 0, batch_axis_size, self.pconfig, promote_to_global=True)
                slps_ixs.append(involutive_kernel(traces, lps, jax.random.split(rng_key, n_samples)))

            slps_ixs = jnp.vstack(slps_ixs)
        
            transition_probs: Dict[int, FloatArray] = dict()
            for a in jnp.unique(slps_ixs):
                transition_probs[int(a)] = jnp.mean(slps_ixs == a)

            return RJMCMCTransitionProbEstimate(transition_probs, n_samples)
        
        def _post_info(result: LogWeightEstimate):
            assert isinstance(result, RJMCMCTransitionProbEstimate)
            jump_probs = ", ".join(f"{a}: {result.transition_probs[a]:.6f}" for a in sorted(result.transition_probs.keys()))
            return f"Estimate log weight for {slp.formatted()}: jump probs = {jump_probs}"
           
        return EstimateLogWeightTask(_task, (rng_key,last_inference_result), post_info=_post_info)

    
    def compute_slp_log_weight(self, log_weight_estimates: Dict[SLP, LogWeightEstimate]) -> Dict[SLP, FloatArray]:
        max_K = max(int(self.get_index(slp.decision_representative)) for slp in log_weight_estimates.keys())
        A = jnp.zeros((max_K+1, max_K+1))
        b = jnp.zeros((max_K+1,))
        for slp, estimate in log_weight_estimates.items():
            current_k = int(self.get_index(slp.decision_representative))
            assert isinstance(estimate, RJMCMCTransitionProbEstimate)
            for next_k, prob in estimate.transition_probs.items():
                if next_k <= max_K:
                    A = A.at[current_k, current_k].add(1)
                    A = A.at[current_k, next_k].add(-1)
                    b = b.at[current_k].add(-jnp.log(prob))
                    b = b.at[next_k].add(jnp.log(prob))
        A = A.at[0,0].add(1)
        
        log_weights = jnp.linalg.solve(A, b)

        result: Dict[SLP, FloatArray] = dict()
        for slp in log_weight_estimates.keys():
            result[slp] = log_weights[int(self.get_index(slp.decision_representative))]

        return result
