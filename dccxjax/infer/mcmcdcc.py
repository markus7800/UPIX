import jax
import jax.numpy as jnp
from typing import Dict, Optional, List, Callable, Any, NamedTuple, Generic, TypeVar, Tuple, cast
from dccxjax.core import SLP, Model, sample_from_prior, slp_from_decision_representative
from ..types import Trace, PRNGKey, FloatArray, IntArray, StackedTrace, StackedTraces, StackedSampleValues, _unstack_sample_data
from dataclasses import dataclass
from .mcmc import MCMCRegime, MCMC, MCMCState, summarise_mcmc_info
from .estimate_Z import estimate_log_Z_for_SLP_from_prior
from time import time
from copy import deepcopy
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from ..utils import broadcast_jaxtree, pprint_dtype_shape_of_tree
from functools import reduce
from .lmh_global import lmh
from .variable_selector import AllVariables, VariableSelector
from .dcc import InferenceResult, LogWeightEstimate, AbstractDCC
from .mcdcc import MCDCC, DCC_COLLECT_TYPE, MCLogWeightEstimate, MCInferenceResult, WeightedSample

__all__ = [
    "MCMCDCC",
    "MCMCInferenceResult",
    "LogWeightEstimateFromPrior",
]

@dataclass
class MCMCInferenceResult(MCInferenceResult[DCC_COLLECT_TYPE]):
    value_tree: None | Tuple[Trace,FloatArray] | DCC_COLLECT_TYPE # either None or Pytree with leading axes (n_samples_per_chain, n_chains,...)
    last_state: MCMCState
    n_chains: int
    n_samples_per_chain: int
    optimised_memory_with_early_return_map: bool
    def combine(self, other: InferenceResult) -> "MCMCInferenceResult":
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
    
    def get_weighted_sample(self, return_map: Callable[[Trace],DCC_COLLECT_TYPE]) -> WeightedSample[DCC_COLLECT_TYPE]:
        if self.value_tree is not None:
            if self.optimised_memory_with_early_return_map:
                # assert isinstance(inference_result.value_tree, DCC_COLLECT_TYPE)
                values: DCC_COLLECT_TYPE = cast(DCC_COLLECT_TYPE, self.value_tree)
            else:
                # assert isinstance(inference_result.value_tree, Trace)
                values = return_map(cast(Tuple[Trace,FloatArray], self.value_tree)[0])

            weighted_samples = WeightedSample(
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
            weighted_samples = WeightedSample(
                StackedSampleValues(values, n_mcmc, n_chains),
                jnp.zeros((n_mcmc,n_chains), float)
            )
        # print(pprint_dtype_shape_of_tree(self.value_tree))
        # print(pprint_dtype_shape_of_tree(weighted_samples.values.data))
        return weighted_samples

@dataclass
class LogWeightEstimateFromPrior(MCLogWeightEstimate):
    log_Z: FloatArray
    ESS: IntArray
    frac_in_support: FloatArray
    n_samples: int
    def combine(self, other: LogWeightEstimate) -> "LogWeightEstimateFromPrior":
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
                    progress_bar=self.verbose >= 1)
        self.inference_method_cache[slp] = mcmc
        return mcmc
    
    def estimate_log_weight(self, slp: SLP, rng_key: PRNGKey) -> LogWeightEstimate:
        log_Z, ESS, frac_in_support = estimate_log_Z_for_SLP_from_prior(slp, self.estimate_weight_n_samples, rng_key)
        if self.verbose >= 2:
            tqdm.write(f"Estimated log weight for {slp.formatted()}: {log_Z.item()} (ESS={ESS.item():,.0f})")
        return LogWeightEstimateFromPrior(log_Z, ESS, frac_in_support, self.estimate_weight_n_samples)

    def run_inference(self, slp: SLP, rng_key: PRNGKey) -> InferenceResult:
        mcmc = self.get_MCMC(slp)
        inference_results = self.inference_results.get(slp, [])
        if len(inference_results) > 0:
            last_result = inference_results[-1]
            assert isinstance(last_result, MCMCInferenceResult)
            init_positions = StackedTrace(last_result.last_state.position, mcmc.n_chains)
            log_prob = last_result.last_state.log_prob
        else:
            init_positions = StackedTrace(broadcast_jaxtree(slp.decision_representative, (mcmc.n_chains,)), mcmc.n_chains)
            log_prob = broadcast_jaxtree(slp.log_prob(slp.decision_representative), (mcmc.n_chains,))
        
        last_state, return_result = mcmc.run(rng_key, init_positions, log_prob, n_samples_per_chain=self.mcmc_n_samples_per_chain)
        if self.verbose >= 2 and self.mcmc_collect_inference_info:
            assert last_state.infos is not None
            info_str = "MCMC Infos:"
            for step, info in enumerate(last_state.infos):
                info_str += f"\n\t Step {step}: {summarise_mcmc_info(info, self.mcmc_n_samples_per_chain)}"
            tqdm.write(info_str)
        
        # TODO: only run MCMC on gpu and handle all result data on CPU
        cpu = jax.devices("cpu")[0]
        return_result = jax.device_put(return_result, cpu)
        # we do not apply return map here, because we want to be able to continue mcmc chain from last state
        return MCMCInferenceResult(return_result, last_state, mcmc.n_chains, self.mcmc_n_samples_per_chain, self.mcmc_optimise_memory_with_early_return_map)