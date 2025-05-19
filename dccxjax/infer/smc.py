import jax
from typing import NamedTuple, Tuple, Set, Dict, Callable, Optional
from ..types import PRNGKey, Trace, FloatArray, IntArray, StackedTrace, BoolArray
from ..core.model_slp import SLP
from .mcmc import MCMCKernel, MCMCState, CarryStats, MCMCRegime, get_mcmc_kernel, InferenceInfos, init_inference_infos_for_chains
from ..utils import broadcast_jaxtree
import jax.numpy as jnp
from abc import ABC, abstractmethod

# from blackjax/smc/resampling.py
def sorted_uniforms(rng_key: PRNGKey, n) -> FloatArray:
    # Credit goes to Nicolas Chopin
    es = jax.random.exponential(rng_key, (n + 1,))
    z = jnp.cumsum(es)
    return z[:-1] / z[-1]

def resampling_multinomial(rng_key: PRNGKey, weights: FloatArray, num_samples: int) -> IntArray:
    n = weights.shape[0]
    linspace = sorted_uniforms(rng_key, num_samples)
    cumsum = jnp.cumsum(weights)
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)

def _systematic_or_stratified(rng_key: PRNGKey, weights: FloatArray, num_samples: int, is_systematic: bool) -> IntArray:
    n = weights.shape[0]
    if is_systematic:
        u = jax.random.uniform(rng_key, ())
    else:
        u = jax.random.uniform(rng_key, (num_samples,))
    cumsum = jnp.cumsum(weights)
    linspace = (jnp.arange(num_samples, dtype=weights.dtype) + u) / num_samples
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n - 1)

def resampling_systematic(rng_key: PRNGKey, weights: FloatArray, num_samples: int) -> IntArray:
    return _systematic_or_stratified(rng_key, weights, num_samples, True)

def resampling_stratified(rng_key: PRNGKey, weights: FloatArray, num_samples: int) -> IntArray:
    return _systematic_or_stratified(rng_key, weights, num_samples, False)

from enum import Enum
class ResampleType(Enum):
    Never = 0
    Adaptive = 1
    Always = 2

class ReweightingType(Enum):
    Bootstrap = 0
    BootstrapStaticPrior = 1
    Guided = 2

class SMCResampling(ABC):
    def __init__(self, resample_fn: Callable[[PRNGKey, FloatArray, int], IntArray], resample_type: ResampleType) -> None:
        self.resample_fn = resample_fn
        self.resample_type = resample_type
    def resample(self, rng_key: PRNGKey, weights: FloatArray, num_samples: int):
        return self.resample_fn(rng_key, weights, num_samples)

def MultinomialResampling(resample_type: ResampleType): return SMCResampling(resampling_multinomial, resample_type)
def StratifiedResampling(resample_type: ResampleType): return SMCResampling(resampling_stratified, resample_type)
def SystematicResampling(resample_type: ResampleType): return SMCResampling(resampling_systematic, resample_type)

class DataAnnealingSchedule(NamedTuple):
    data_annealing: Dict[str,BoolArray]
    n_steps: int

class TemperetureSchedule(NamedTuple):
    temperature: FloatArray
    n_steps: int

class SMCState(NamedTuple):
    particles: Trace
    log_particle_weights: FloatArray
    ta_log_likelihood: FloatArray | None
    ta_log_prob: FloatArray # tempered / annealed log prob
    mcmc_info: InferenceInfos | None

class SMCStepData(NamedTuple):
    rng_key: PRNGKey
    temperature: FloatArray
    data_annealing: Dict[str, BoolArray]

def get_smc_step(slp: SLP, n_particles: int, reweighting_type: ReweightingType, resampling: SMCResampling, rejuvinate_kernel: MCMCKernel[None]):

    def do_resample_fn(particles: Trace, log_prob: FloatArray, log_particle_weights: FloatArray, rng_key: PRNGKey):
        log_particle_weigths_sum = jax.scipy.special.logsumexp(log_particle_weights)
        weights = jax.lax.exp(log_particle_weights - log_particle_weigths_sum)
        ixs = resampling.resample(rng_key, weights, n_particles)
        particles = jax.tree.map(lambda v: v[ixs,...], particles)
        return particles, jax.lax.broadcast(log_particle_weigths_sum - jax.lax.log(float(n_particles)), (n_particles,)) # proper weighting

    def no_resample_fn(particles: Trace, log_prob: FloatArray, log_weights: FloatArray, rng_key: PRNGKey):
        return particles, log_prob

    @jax.jit
    def smc_step(smc_state: SMCState, step_data: SMCStepData) -> Tuple[SMCState, FloatArray]:
        rejuvinate_key, resample_key = jax.random.split(step_data.rng_key)
       
        # density with respect to current tempering / annealing
        a_log_prior_current, a_log_likelihood_current, _ = jax.vmap(slp._log_prior_likeli_pathcond, in_axes=(0,None))(smc_state.particles, step_data.data_annealing)
        ta_log_likelihood_current = step_data.temperature * a_log_likelihood_current
        ta_log_prob_current = a_log_prior_current + ta_log_likelihood_current
        particles_current = smc_state.particles

        # reweight
        # In bootstrapping, we compute
        # p(y_{1:k}, x_{1:k}) / (p(y_{1:k-1}, x_{1:k-1}) * p(x_k | x_{1:k-1})) = p(y_k | x_k)
        # i.e. when we increase latent dimension, we "instantiate" from prior (updating data annealing mask of latent variable)
        # In guided, we compute
        # p(y_{1:k}, x_{1:k}) / (p(y_{1:k-1}, x_{1:k-1}) * q(x_k | y_{1:k}, x_{1:k-1})) = p(y_k | x_k) * p(x_k | x_{1:k-1}) / q(x_k | y_{1:k}, x_{1:k-1}))
        if reweighting_type == ReweightingType.BootstrapStaticPrior:
            # data annealing only used for observed variables
            log_w_hat = ta_log_prob_current - smc_state.ta_log_prob # = ta_log_likelihood_current - smc_state.ta_log_likelihood
        elif reweighting_type == ReweightingType.Bootstrap:
            # a_log_prior_current may be different from (smc_state.ta_log_prob - smc_state.ta_log_likelihood) due to data annealing
            assert smc_state.ta_log_likelihood is not None
            log_w_hat = ta_log_likelihood_current - smc_state.ta_log_likelihood
        else:
            raise NotImplementedError
        
        log_w_hat = ta_log_prob_current - smc_state.ta_log_prob # log [ p_k(phi_{k-1}) / p_{k-1}(phi_{k-1}) ]
        log_particle_weights = smc_state.log_particle_weights + log_w_hat
        
        log_ess = jax.scipy.special.logsumexp(log_particle_weights)  * 2 - jax.scipy.special.logsumexp(log_particle_weights*2)

        # resample (only llorente2023 puts it after rejuvinate)
        # if resampling.resample_type != ResampleType.Never:
        #     if resampling.resample_type == ResampleType.Always:
        #         particles_current, ta_log_prob_current = do_resample_fn(particles_current, ta_log_prob_current, log_particle_weights, resample_key)
        #     elif resampling.resample_type == ResampleType.Adaptive:
        #         particles_current, ta_log_prob_current = jax.lax.cond(log_ess < jax.lax.log(n_particles / 2.0), do_resample_fn, no_resample_fn, particles_current, ta_log_prob_current, log_particle_weights, resample_key)

        # rejuvinate
        current_mcmc_state = MCMCState(
            jnp.array(0, int),
            step_data.temperature,
            step_data.data_annealing,
            particles_current,
            ta_log_prob_current,
            CarryStats(), # to support carry stats we would have to recompute here (e.g. unconstrained_log_prob)
            smc_state.mcmc_info
        )
        next_mcmc_state, _ = rejuvinate_kernel(current_mcmc_state, rejuvinate_key)
        particles = next_mcmc_state.position
        ta_log_prob = next_mcmc_state.log_prob

        # resample (only llorente2023 puts it after rejuvinate)
        if resampling.resample_type != ResampleType.Never:
            if resampling.resample_type == ResampleType.Always:
                particles, ta_log_prob = do_resample_fn(particles, ta_log_prob, log_particle_weights, resample_key)
            elif resampling.resample_type == ResampleType.Adaptive:
                particles, ta_log_prob = jax.lax.cond(log_ess < jax.lax.log(n_particles / 2.0), do_resample_fn, no_resample_fn, particles_current, ta_log_prob_current, log_particle_weights, resample_key)


        ta_log_likelihood = jax.vmap(slp.log_likelihood, in_axes=(0,None))(particles, step_data.data_annealing) if reweighting_type == ReweightingType.Bootstrap else None

        return SMCState(particles, log_particle_weights, ta_log_likelihood, ta_log_prob, next_mcmc_state.infos), jax.lax.exp(log_ess)

    return smc_step

class SMC:
    def __init__(self,
                 slp: SLP,
                 n_particles: int,
                 tempereture_schedule: Optional[TemperetureSchedule],
                 data_annealing_schedule: Optional[DataAnnealingSchedule],
                 reweighting_type: ReweightingType,
                 resampling: SMCResampling,
                 rejuvination_regime: MCMCRegime,
                 *,
                 collect_inference_info: bool = False,
                 progress_bar: bool = False) -> None:
        
        self.slp = slp
        self.n_particles = n_particles
        self.progress_bar = progress_bar

        rejuvination_kernel, rejuv_init_carry_stat_names = get_mcmc_kernel(slp, rejuvination_regime, collect_inference_info=collect_inference_info, vectorised=True, return_map=lambda _: None)
        assert len(rejuv_init_carry_stat_names) == 0, "CarryStats in MCMC currently not supported."

        self.rejuvination_regime = rejuvination_regime
        self.rejuvination_kernel: MCMCKernel[None] = rejuvination_kernel
        self.collect_inference_info = collect_inference_info

        if tempereture_schedule is None:
            assert data_annealing_schedule is not None
            self.tempereture_schedule = TemperetureSchedule(jnp.ones((data_annealing_schedule.n_steps,), float), data_annealing_schedule.n_steps)
        else:
            self.tempereture_schedule = tempereture_schedule
        if data_annealing_schedule is None:
            assert tempereture_schedule is not None
            self.data_annealing_schedule = DataAnnealingSchedule(dict(), tempereture_schedule.n_steps)
        else:
            self.data_annealing_schedule = data_annealing_schedule
        assert self.tempereture_schedule.n_steps == self.data_annealing_schedule.n_steps
        self.n_steps = self.data_annealing_schedule.n_steps

        self.reweighting_type = reweighting_type
        self.resampling = resampling

        self.smc_step = get_smc_step(slp, n_particles, reweighting_type, resampling, rejuvination_kernel)

    def run(self, rng_key: PRNGKey, particles: StackedTrace, log_prob: Optional[FloatArray] = None):
        # no tempering / annealing weights, we require that this is proper weighting for input particles
        if log_prob is None:
            log_prior, _, path_condition = jax.vmap(self.slp._log_prior_likeli_pathcond, in_axes=(0,None))(particles.data, dict())
            log_particle_weights = jax.lax.select(path_condition, log_prior, jax.lax.zeros_like_array(log_prior))
            log_prob = log_prior
        else:
            log_particle_weights = log_prob
            
        n_particles = particles.n_samples()

        smc_keys = jax.random.split(rng_key, self.n_steps)

        mcmc_infos = init_inference_infos_for_chains(self.rejuvination_regime, n_particles) if self.collect_inference_info else None

        ta_log_likelihood = jax.lax.zeros_like_array(log_prob) if self.reweighting_type == ReweightingType.Bootstrap else None
        last_state, ess = jax.lax.scan(self.smc_step,
            SMCState(particles.data, log_particle_weights, ta_log_likelihood, log_prob, mcmc_infos),
            SMCStepData(smc_keys, self.tempereture_schedule.temperature, self.data_annealing_schedule.data_annealing)
        )

        return last_state, ess 