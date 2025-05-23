import jax
import jax.numpy as jnp
from typing import Dict, Optional, List, Callable, Any, NamedTuple, Generic, TypeVar, Tuple, cast
from dccxjax.core import SLP, Model, sample_from_prior, slp_from_decision_representative
from ..types import Trace, PRNGKey, FloatArray, IntArray, StackedTrace, StackedTraces, StackedSampleValues, _unstack_sample_data
from dataclasses import dataclass
from .smc import SMC, DataAnnealingSchedule, TemperetureSchedule, ReweightingType, StratifiedResampling, ResampleType, ResampleTime
from .mcmc import MCMCRegime, summarise_mcmc_infos, MCMC
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
from textwrap import indent

__all__ = [
    "SMCDCC"
]

@dataclass
class SMCInferenceResult(MCInferenceResult[DCC_COLLECT_TYPE]):
    particles: Tuple[Trace,FloatArray] | DCC_COLLECT_TYPE
    log_particle_weight: FloatArray
    n_particles: int
    optimised_memory_with_early_return_map: bool
    def combine(self, other: InferenceResult) -> "SMCInferenceResult":
        if not isinstance(other, SMCInferenceResult):
            raise TypeError
        
        # we put iterations on leading axes, i.e. shape = (#repeats smc, n_particles, ...)
        assert self.n_particles == other.n_particles
        particles_1, log_particle_weight_1 = broadcast_jaxtree((self.particles, self.log_particle_weight), (1,)) if len(self.log_particle_weight.shape) == 1 else (self.particles, self.log_particle_weight)
        particles_2, log_particle_weight_2 = broadcast_jaxtree((other.particles, other.log_particle_weight), (1,)) if len(other.log_particle_weight.shape) == 1 else (other.particles, other.log_particle_weight)

        particles = jax.tree.map(lambda x, y: jax.lax.concatenate((x, y), 0), particles_1, particles_2)
        log_particle_weight = jax.lax.concatenate((log_particle_weight_1, log_particle_weight_2), 0)

        assert self.optimised_memory_with_early_return_map == other.optimised_memory_with_early_return_map

        return SMCInferenceResult(particles, log_particle_weight, self.n_particles+other.n_particles, self.optimised_memory_with_early_return_map)
    
    def get_weighted_sample(self, return_map: Callable[[Trace],DCC_COLLECT_TYPE]) -> WeightedSample[DCC_COLLECT_TYPE]:
        # shape = (#repeats smc, n_particles, ...)
        # we add axis when combining, if we have not combined anything, we have to add axis now
        particles, log_particle_weight = broadcast_jaxtree((self.particles, self.log_particle_weight), (1,)) if len(self.log_particle_weight.shape) == 1 else (self.particles, self.log_particle_weight)
        n_smc = log_particle_weight.shape[0]

        values: DCC_COLLECT_TYPE = cast(DCC_COLLECT_TYPE,particles) if self.optimised_memory_with_early_return_map else return_map(cast(Trace,particles))
        
        # normalise log_particle_weights on second axis
        # each smc run is treated independently
        log_weights = log_particle_weight - jax.scipy.special.logsumexp(log_particle_weight, axis=1)

        weighted_samples = WeightedSample(
            StackedSampleValues(values, n_smc, self.n_particles),
            log_weights
        )
        return weighted_samples


@dataclass
class LogWeightEstimateFromSMC(MCLogWeightEstimate):
    log_Z: FloatArray
    ESS: IntArray
    n_particles: int
    def combine(self, other: LogWeightEstimate) -> "LogWeightEstimateFromSMC":
        assert isinstance(other, LogWeightEstimateFromSMC)
        n_combined_particles = self.n_particles + other.n_particles
        a = self.n_particles / n_combined_particles
        
        log_Z = jax.numpy.logaddexp(self.log_Z + jax.lax.log(a), other.log_Z + jax.lax.log(1 - a))

        return LogWeightEstimateFromSMC(log_Z, self.ESS + other.ESS, n_combined_particles)
    
    def get_estimate(self):
        return self.log_Z
    
class SMCDCC(MCDCC[DCC_COLLECT_TYPE]):
    def __init__(self, model: Model, return_map: Callable[[Trace], DCC_COLLECT_TYPE] = lambda trace: trace, *ignore, verbose=0, **config_kwargs) -> None:
        super().__init__(model, return_map, *ignore, verbose=verbose, **config_kwargs)

        self.smc_collect_inference_info: bool = self.config.get("smc_collect_inference_info", True)
        self.smc_optimise_memory_with_early_return_map: bool = self.config.get("smc_optimise_memory_with_early_return_map", False)
        self.smc_n_particles: int = self.config.get("smc_n_particles", 100)
        self.smc_prior_mcmc_n_steps: int = self.config.get("smc_prior_mcmc_n_steps", 100)
        self.smc_rejuvination_attempts: int = self.config.get("smc_rejuvination_attempts", 1)
        self.est_path_log_prob_n_samples: int = self.config.get("est_path_log_prob_n_samples", 10_000)

    def get_SMC_tempering_schedule(self, slp: SLP) -> Optional[TemperetureSchedule]:
        return None
    
    def get_SMC_data_annealing_schedule(self, slp: SLP) -> Optional[DataAnnealingSchedule]:
        return None
    
    @abstractmethod
    def get_SMC_rejuvination_kernel(self, slp: SLP) -> MCMCRegime:
        raise NotImplementedError

    def get_SMC(self, slp: SLP) -> SMC:
        if slp in self.inference_method_cache:
            smc = self.inference_method_cache[slp]
            assert isinstance(smc, SMC)
            return smc
            
        smc = SMC(
            slp,
            self.smc_n_particles,
            self.get_SMC_tempering_schedule(slp),
            self.get_SMC_data_annealing_schedule(slp),
            ReweightingType.BootstrapStaticPrior,
            StratifiedResampling(ResampleType.Adaptive, ResampleTime.BeforeMove),
            self.get_SMC_rejuvination_kernel(slp),
            self.smc_rejuvination_attempts,
            collect_inference_info=self.smc_collect_inference_info,
            progress_bar=True
        )

        self.inference_method_cache[slp] = smc
        return smc

    def produce_samples_from_prior(self, slp: SLP, rng_key: PRNGKey) -> Tuple[StackedTrace, Optional[FloatArray]]:
        # by default reuse rejuvination kernel
        smc = self.get_SMC(slp)
        smc.rejuvination_kernel
        mcmc = MCMC(
            slp,
            smc.rejuvination_regime,
            n_chains=smc.n_particles,
            reuse_kernel=smc.rejuvination_kernel,
            reuse_kernel_init_carry_stat_names=smc.rejuvination_kernel_init_carry_stat_names,
            data_annealing = smc.data_annealing_schedule.prior_mask() if smc.data_annealing_schedule is not None else dict(),
            temperature = jnp.array(0.,float),
            collect_inference_info=smc.collect_inference_info,
            progress_bar=True
        )
        init_positions = StackedTrace(broadcast_jaxtree(slp.decision_representative, (mcmc.n_chains,)), mcmc.n_chains)

        last_state, _ = mcmc.run(rng_key, init_positions, n_samples_per_chain=self.smc_prior_mcmc_n_steps)

        if self.verbose >= 2 and self.smc_collect_inference_info:
            assert last_state.infos is not None
            info_str = "Prior MCMC Infos:\n"
            info_str += indent(summarise_mcmc_infos(last_state.infos, self.smc_prior_mcmc_n_steps), "\t")
            tqdm.write(info_str)

        return StackedTrace(last_state.position, mcmc.n_chains), last_state.log_prob

    def run_inference(self, slp: SLP, rng_key: PRNGKey) -> InferenceResult:
        smc = self.get_SMC(slp)
        
        prior_key, smc_key = jax.random.split(rng_key)

        init_positions, init_log_prob = self.produce_samples_from_prior(slp, prior_key)

        last_state, ess = smc.run(smc_key, init_positions, init_log_prob)
        if self.verbose >= 2:
            if self.smc_collect_inference_info:
                assert last_state.mcmc_infos is not None
                info_str = "Rejuvination Infos:\n"
                info_str += indent(summarise_mcmc_infos(last_state.mcmc_infos, smc.n_steps*smc.rejuvination_attempts), "\t")
                tqdm.write(info_str)
        
        if self.smc_optimise_memory_with_early_return_map:
            return_result = self.return_map(last_state.particles)
        else:
            return_result = last_state.particles
        return SMCInferenceResult(return_result, last_state.log_particle_weights, smc.n_particles, self.smc_optimise_memory_with_early_return_map)
    
    def estimate_path_log_prob(self, slp: SLP, rng_key: PRNGKey) -> FloatArray:
        _, _, frac_in_support = estimate_log_Z_for_SLP_from_prior(slp, self.est_path_log_prob_n_samples, rng_key)
        return jax.lax.log(frac_in_support)

    def estimate_log_weight(self, slp: SLP, rng_key: PRNGKey) -> LogWeightEstimate:
        inference_results = self.inference_results.get(slp, [])
        if len(inference_results) > 0:
            path_log_prob = self.estimate_path_log_prob(slp, rng_key)

            last_result = inference_results[-1]
            assert isinstance(last_result, SMCInferenceResult)
            assert len(last_result.log_particle_weight.shape) == 1, "Attempted to get log_weight from combined result"
            log_Z = jax.scipy.special.logsumexp(last_result.log_particle_weight)
            log_ess = log_Z * 2 - jax.scipy.special.logsumexp(last_result.log_particle_weight * 2)
            ESS = jax.lax.exp(log_ess)
            if self.verbose >= 2:
                tqdm.write(f"Estimated log weight for {slp.formatted()}: {log_Z.item()} (ESS={ESS.item():_.0f})")
            return LogWeightEstimateFromSMC(log_Z + path_log_prob, ESS, last_result.n_particles)
        else:
            raise Exception("In SMCDCC we should perform one run of SMC before estimate_log_weight to reuse estimate")
    