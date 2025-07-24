import jax
import jax.numpy as jnp
from typing import Dict, Optional, List, Callable, Any, NamedTuple, Generic, TypeVar, Tuple, cast
from dccxjax.core import SLP, Model
from dccxjax.types import Trace, PRNGKey, FloatArray, IntArray, StackedTrace, StackedTraces, StackedSampleValues, _unstack_sample_data
from dataclasses import dataclass
from dccxjax.infer.mcmc.mcmc_core import MCMCRegime, MCMC, MCMCState, summarise_mcmc_infos
from dccxjax.infer.importance_sampling import estimate_log_Z_for_SLP_from_prior
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from dccxjax.utils import broadcast_jaxtree, pprint_dtype_shape_of_tree
from dccxjax.infer.dcc.abstract_dcc import InferenceTask, EstimateLogWeightTask, InferenceResult, LogWeightEstimate, AbstractDCC, ParallelisationType
from dccxjax.infer.dcc.mc_dcc import MCDCC, DCC_COLLECT_TYPE, MCLogWeightEstimate, MCInferenceResult, LogWeightedSample
from textwrap import indent
import time

__all__ = [
    "MCMCDCC",
    "MCMCInferenceResult",
    "LogWeightEstimateFromPrior",
]

@jax.tree_util.register_dataclass
@dataclass
class MCMCInferenceResult(MCInferenceResult[DCC_COLLECT_TYPE]):
    value_tree: None | Tuple[Trace,FloatArray] | DCC_COLLECT_TYPE # either None or Pytree with leading axes (n_samples_per_chain, n_chains,...)
    last_state: MCMCState
    n_chains: int
    n_samples_per_chain: int
    optimised_memory_with_early_return_map: bool
    def combine_results(self, other: InferenceResult) -> "MCMCInferenceResult":
        if not isinstance(other, MCMCInferenceResult):
            raise TypeError
        assert self.n_chains == other.n_chains
        if self.value_tree is None:
            assert other.value_tree is None
            value_tree = None
        else:
            assert other.value_tree is not None
            # shape = (n_samples_per_chain, n_chain, ...)
            value_tree = jax.tree.map(lambda x, y: jax.lax.concatenate((x, y), 0), self.value_tree, other.value_tree)
        # shape = (n_chain, ...)
        # we add axis to last_state if needed
       
        last_state_1 = broadcast_jaxtree(self.last_state, (1,)) if self.last_state.iteration.shape == () else self.last_state
        last_state_2 = broadcast_jaxtree(other.last_state, (1,)) if other.last_state.iteration.shape == () else other.last_state
        last_state = jax.tree.map(lambda x, y: jax.lax.concatenate((x, y), 0), last_state_1, last_state_2)

        assert self.optimised_memory_with_early_return_map == other.optimised_memory_with_early_return_map

        return MCMCInferenceResult(value_tree, last_state, self.n_chains, self.n_samples_per_chain + other.n_samples_per_chain, self.optimised_memory_with_early_return_map)
    
    def get_weighted_sample(self, return_map: Callable[[Trace],DCC_COLLECT_TYPE]) -> LogWeightedSample[DCC_COLLECT_TYPE]:
        if self.value_tree is not None:
            if self.optimised_memory_with_early_return_map:
                # assert isinstance(inference_result.value_tree, DCC_COLLECT_TYPE)
                values: DCC_COLLECT_TYPE = cast(DCC_COLLECT_TYPE, self.value_tree)
            else:
                # assert isinstance(inference_result.value_tree, Trace)
                values = return_map(cast(Tuple[Trace,FloatArray], self.value_tree)[0])

            weighted_samples = LogWeightedSample(
                StackedSampleValues(values, self.n_samples_per_chain, self.n_chains),
                jnp.zeros((self.n_samples_per_chain, self.n_chains), float)
            )
        else:
            # number of samples per chain = number of times MCMC was performed for SLP
            # because we only stored last state
            n_chains = self.n_chains
            n_mcmc = self.last_state.iteration.size
            # we add axis when combining, if we have not combined anything, we have to add axis now
            last_state = broadcast_jaxtree(self.last_state, (1,)) if self.last_state.iteration.shape == () else self.last_state
            values = return_map(last_state.position)
            weighted_samples = LogWeightedSample(
                StackedSampleValues(values, n_mcmc, n_chains),
                jnp.zeros((n_mcmc,n_chains), float)
            )
        # print(pprint_dtype_shape_of_tree(self.value_tree))
        # print(pprint_dtype_shape_of_tree(weighted_samples.values.data))
        return weighted_samples

@jax.tree_util.register_dataclass
@dataclass
class LogWeightEstimateFromPrior(MCLogWeightEstimate):
    log_Z: FloatArray
    ESS: IntArray
    frac_in_support: FloatArray
    n_samples: int
    def combine_estimates(self, other: LogWeightEstimate) -> "LogWeightEstimateFromPrior":
        assert isinstance(other, LogWeightEstimateFromPrior)
        n_combined_samples = self.n_samples + other.n_samples
        a = self.n_samples / n_combined_samples
        
        log_Z = jax.numpy.logaddexp(self.log_Z + jax.lax.log(a), other.log_Z + jax.lax.log(1 - a))
        frac_in_support = self.frac_in_support * a + other.frac_in_support * (1 - a)

        return LogWeightEstimateFromPrior(log_Z, self.ESS + other.ESS, frac_in_support, n_combined_samples)
    
    def get_estimate(self):
        return self.log_Z

class MCMCDCC(MCDCC[DCC_COLLECT_TYPE]):
    def __init__(self, model: Model, return_map: Callable[[Trace], DCC_COLLECT_TYPE] = lambda trace: trace, *ignore, verbose=0, **config_kwargs) -> None:
        super().__init__(model, return_map, *ignore, verbose=verbose, **config_kwargs)

        self.mcmc_collect_inference_info: bool = self.config.get("mcmc_collect_inference_info", True)
        self.mcmc_collect_for_all_traces: bool = self.config.get("mcmc_collect_for_all_traces", True)
        self.mcmc_optimise_memory_with_early_return_map: bool = self.config.get("mcmc_optimise_memory_with_early_return_map", False)
        self.mcmc_n_chains: int = self.config.get("mcmc_n_chains", 4)
        self.mcmc_n_samples_per_chain: int = self.config.get("mcmc_n_samples_per_chain", 1_000)

        if self.max_iterations > 1:
            assert not self.mcmc_optimise_memory_with_early_return_map

    @abstractmethod
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        raise NotImplementedError

    def get_MCMC(self, slp: SLP) -> MCMC:
        if slp in self.inference_method_cache:
            mcmc = self.inference_method_cache[slp]
            assert isinstance(mcmc, MCMC)
            return mcmc
        regime = self.get_MCMC_inference_regime(slp)
        def mcmc_return_map(state: MCMCState):
            if self.mcmc_collect_for_all_traces:
                if self.mcmc_optimise_memory_with_early_return_map:
                    return self.return_map(state.position)
                else:
                    return (state.position, state.log_prob)
            else:
                return None
        mcmc = MCMC(slp, regime, self.mcmc_n_chains,
                    collect_inference_info=self.mcmc_collect_inference_info,
                    return_map=mcmc_return_map,
                    show_progress=self.verbose >= 1 and self.parallelisation.type == ParallelisationType.Sequential,
                    shared_progressbar=self.shared_progress_bar)
        self.inference_method_cache[slp] = mcmc
        return mcmc
    
    def make_estimate_log_weight_task(self, slp: SLP, rng_key: PRNGKey) -> EstimateLogWeightTask:
        def _f(rng_key: PRNGKey):
            log_Z, ESS, frac_in_support = estimate_log_Z_for_SLP_from_prior(slp, self.estimate_weight_n_samples, rng_key)
            return LogWeightEstimateFromPrior(log_Z, ESS, frac_in_support, self.estimate_weight_n_samples)
        
        def _post_info(result: LogWeightEstimate):
            assert isinstance(result, LogWeightEstimateFromPrior)
            if self.verbose >= 2:
                return f"Estimated log weight for {slp.formatted()}: {result.log_Z.item()} (ESS={result.ESS.item():_.0f})"
            return ""
        
        return EstimateLogWeightTask(_f, (rng_key,), post_info=_post_info)

    def make_inference_task(self, slp: SLP, rng_key: PRNGKey) -> InferenceTask:
        mcmc = self.get_MCMC(slp)
        inference_results = self.inference_results.get(slp, [])

        def _post_info(result: InferenceResult):
            assert isinstance(result, MCMCInferenceResult)
            if self.verbose >= 2 and self.mcmc_collect_inference_info:
                assert result.last_state.infos is not None
                info_str = f"MCMC Infos for {slp.formatted()}:\n"
                info_str += indent(summarise_mcmc_infos(result.last_state.infos, self.mcmc_n_samples_per_chain), "\t")
                return info_str
            return ""
        
        if len(inference_results) > 0:
            last_result = inference_results[-1]
            assert isinstance(last_result, MCMCInferenceResult)
            # init_positions = StackedTrace(last_result.last_state.position, mcmc.n_chains)
            # init_log_prob = last_result.last_state.log_prob
            # TODO: test this
            def _task_continue(rng_key: PRNGKey, state: MCMCState):
                last_state, return_result = mcmc.continue_run(rng_key, state, n_samples_per_chain=self.mcmc_n_samples_per_chain)
                return MCMCInferenceResult(return_result, last_state, mcmc.n_chains, self.mcmc_n_samples_per_chain, self.mcmc_optimise_memory_with_early_return_map)
            
            return InferenceTask(_task_continue, (rng_key, last_result.last_state), post_info=_post_info)
        else:
            def _task_run(rng_key: PRNGKey):
                init_positions = StackedTrace(broadcast_jaxtree(slp.decision_representative, (mcmc.n_chains,)), mcmc.n_chains)
                init_log_prob = broadcast_jaxtree(slp.log_prob(slp.decision_representative), (mcmc.n_chains,))

                last_state, return_result = mcmc.run(rng_key, init_positions, init_log_prob, n_samples_per_chain=self.mcmc_n_samples_per_chain)
                
                # we do not apply return map here, because we want to be able to continue mcmc chain from last state
                return MCMCInferenceResult(return_result, last_state, mcmc.n_chains, self.mcmc_n_samples_per_chain, self.mcmc_optimise_memory_with_early_return_map)
            return InferenceTask(_task_run, (rng_key, ), post_info=_post_info)

        