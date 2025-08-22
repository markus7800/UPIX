import jax
from typing import NamedTuple, Tuple, Set, Dict, Callable, Optional
from dccxjax.types import PRNGKey, Trace, FloatArray, IntArray, StackedTrace, BoolArray
from dccxjax.core.model_slp import SLP, AnnealingMask
from dccxjax.infer.mcmc.mcmc_core import MCMCKernel, MCMCState, CarryStats, MCMCRegime, get_mcmc_kernel, InferenceInfos, init_inference_infos_for_chains
from dccxjax.utils import broadcast_jaxtree
import jax.numpy as jnp
from abc import ABC, abstractmethod
from jax.flatten_util import ravel_pytree
import jax.experimental
from dccxjax.progress_bar import _add_progress_bar, ProgressbarManager
from tqdm.auto import tqdm
from dccxjax.parallelisation import ParallelisationConfig, ParallelisationType, VectorisationType, vectorise_scan, parallel_run, smap_vmap, SHARDING_AXIS

__all__ = [
    "ResampleType",
    "ResampleTime",
    "ReweightingType",
    "SMCResampling",
    "MultinomialResampling",
    "StratifiedResampling",
    "SystematicResampling",
    "DataAnnealingSchedule",
    "data_annealing_schedule_from_range",
    "TemperetureSchedule",
    "tempering_schedule_from_array",
    "tempering_schedule_from_sigmoid",
    "SMC",
    "get_log_Z_ESS",
    "get_Z_ESS",
    "normalise_log_weights"
]

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
    Adaptive = 0
    Always = 1
class ResampleTime(Enum):
    BeforeMove = 0
    AfterMove = 1
    Never = 2

class ReweightingType(Enum):
    Bootstrap = 0
    BootstrapStaticPrior = 1
    Guided = 2

class SMCResampling(ABC):
    def __init__(self, resample_fn: Callable[[PRNGKey, FloatArray, int], IntArray], resample_type: ResampleType, resample_time: ResampleTime) -> None:
        self.resample_fn = resample_fn
        self.resample_type = resample_type
        self.resample_time = resample_time
    def resample(self, rng_key: PRNGKey, weights: FloatArray, num_samples: int):
        return self.resample_fn(rng_key, weights, num_samples)

def MultinomialResampling(resample_type: ResampleType, resample_time: ResampleTime = ResampleTime.BeforeMove): return SMCResampling(resampling_multinomial, resample_type, resample_time)
def StratifiedResampling(resample_type: ResampleType, resample_time: ResampleTime = ResampleTime.BeforeMove): return SMCResampling(resampling_stratified, resample_type, resample_time)
def SystematicResampling(resample_type: ResampleType, resample_time: ResampleTime = ResampleTime.BeforeMove): return SMCResampling(resampling_systematic, resample_type, resample_time)
def NoResampling(): return SMCResampling(resampling_systematic, ResampleType.Adaptive, ResampleTime.Never)


class DataAnnealingSchedule(NamedTuple):
    data_annealing: AnnealingMask
    n_steps: int
    def prior_mask(self) -> AnnealingMask:
        prior_data_annealing: AnnealingMask = dict()
        if len(self.data_annealing) > 0:
            prior_data_annealing = jax.tree.map(lambda v: jax.numpy.zeros_like(v[0,...]), self.data_annealing)
            # comment out to be jittable
            # masks, _ = ravel_pytree(prior_data_annealing)
            # assert not masks.any()
        return prior_data_annealing

def data_annealing_schedule_from_range(addr_to_range: Dict[str,range]) -> DataAnnealingSchedule:
    data_annealing: AnnealingMask = dict()
    n_steps = 0
    for addr, r in addr_to_range.items():
        ixs = jnp.arange(0, r.stop)
        masks = []
        for i in r:
            masks.append(ixs < i)
        if max(r) != r.stop:
            masks.append(jnp.full((r.stop,), True))
        data_annealing[addr] = jnp.vstack(masks, bool)
        if n_steps == 0:
            n_steps = len(masks)
        else:
            assert n_steps == len(masks)
    assert n_steps > 0
    return DataAnnealingSchedule(data_annealing, n_steps)

class TemperetureSchedule(NamedTuple):
    temperature: FloatArray
    n_steps: int


def sigmoid(z):
    return 1/(1 + jnp.exp(-z))
def tempering_schedule_from_array(arr: FloatArray):
    assert len(arr.shape) == 1
    return TemperetureSchedule(arr, arr.shape[0])

def tempering_schedule_from_sigmoid(linspace: FloatArray):
    arr = sigmoid(linspace).at[-1].set(1.)
    return tempering_schedule_from_array(arr)

class SMCState(NamedTuple):
    iteration: IntArray
    particles: Trace
    log_particle_weights: FloatArray
    ta_log_likelihood: FloatArray | None
    ta_log_prob: FloatArray # tempered / annealed log prob
    mcmc_infos: InferenceInfos | None

class SMCStepData(NamedTuple):
    rng_key: PRNGKey
    temperature: FloatArray
    data_annealing: AnnealingMask

from functools import partial
def get_smc_step(slp: SLP, reweighting_type: ReweightingType, resampling: SMCResampling, rejuvinate_kernel: MCMCKernel[None], rejuvination_attempts: int, vectorisation: str) -> Callable[[SMCState, SMCStepData],Tuple[SMCState,FloatArray]]:
    assert vectorisation in ("vmap", "smap")
    if vectorisation == "vmap":
        _vmap = partial(jax.vmap, in_axes=(0,None), out_axes=0)
    else:
        _vmap = partial(smap_vmap, axis_name=SHARDING_AXIS, in_axes=(0,None), out_axes=0)
        
    def do_resample_fn(particles: Trace, log_prob: FloatArray, log_particle_weights: FloatArray, rng_key: PRNGKey):
        log_particle_weigths_sum = jax.scipy.special.logsumexp(log_particle_weights)
        weights = jax.lax.exp(log_particle_weights - log_particle_weigths_sum)
        n_particles = weights.size
        ixs = resampling.resample(rng_key, weights, n_particles)
        particles = jax.tree.map(lambda v: v[ixs,...], particles)
        log_prob = log_prob[ixs]
        return particles, log_prob, jax.lax.broadcast(log_particle_weigths_sum - jax.lax.log(float(n_particles)), (n_particles,)) # proper weighting

    def no_resample_fn(particles: Trace, log_prob: FloatArray, log_particle_weights: FloatArray, rng_key: PRNGKey):
        return particles, log_prob, log_particle_weights

    def maybe_resample(particles: Trace, log_prob: FloatArray, log_particle_weights: FloatArray, log_ess: FloatArray, rng_key: PRNGKey):
        if resampling.resample_type == ResampleType.Always:
            return do_resample_fn(particles, log_prob, log_particle_weights, rng_key)
        elif resampling.resample_type == ResampleType.Adaptive:
            n_particles = log_particle_weights.size
            return jax.lax.cond(log_ess < jax.lax.log(n_particles / 2.0), do_resample_fn, no_resample_fn, particles, log_prob, log_particle_weights, rng_key)
        else:
            raise Exception



    @jax.jit
    def smc_step(smc_state: SMCState, step_data: SMCStepData) -> Tuple[SMCState, FloatArray]:
        rejuvinate_key, resample_key = jax.random.split(step_data.rng_key)

        # density with respect to current tempering / annealing
        a_log_prior_current, a_log_likelihood_current, _ = _vmap(slp._log_prior_likeli_pathcond)(smc_state.particles, step_data.data_annealing)
        ta_log_likelihood_current = step_data.temperature * a_log_likelihood_current
        ta_log_prob_current = a_log_prior_current + ta_log_likelihood_current
        particles_current = smc_state.particles

        # reweight log_w_hat = log [ p_k(phi_{k-1}) / p_{k-1}(phi_{k-1}) ]
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
        

        log_particle_weights = smc_state.log_particle_weights + log_w_hat
        
        # jax.debug.print("old_weights={o}\nta_log_prob_current={a}\nsmc_state.ta_log_prob={b}\nlog_w_hat={c}\nnew_weights={n}",  o=smc_state.log_particle_weights, a=ta_log_prob_current, b=smc_state.ta_log_prob, c=log_w_hat, n=log_particle_weights)

        log_ess = jax.scipy.special.logsumexp(log_particle_weights) * 2 - jax.scipy.special.logsumexp(log_particle_weights*2)


        # resample (imo preferrable because duplicated positions will be rejuvinated)
        if resampling.resample_time == ResampleTime.BeforeMove:
            particles_current, ta_log_prob_current, log_particle_weights = maybe_resample(particles_current, ta_log_prob_current, log_particle_weights, log_ess, resample_key)
        

        # rejuvinate
        current_mcmc_state = MCMCState(
            jnp.array(0, int),
            step_data.temperature,
            step_data.data_annealing,
            particles_current,
            ta_log_prob_current,
            CarryStats(), # to support carry stats we would have to recompute here (e.g. unconstrained_log_prob)
            smc_state.mcmc_infos
        )
        if rejuvination_attempts == 1:
            next_mcmc_state, _ = rejuvinate_kernel(current_mcmc_state, rejuvinate_key)
        else:
            next_mcmc_state, _ = jax.lax.scan(rejuvinate_kernel, current_mcmc_state, jax.random.split(rejuvinate_key, rejuvination_attempts))
        particles = next_mcmc_state.position
        ta_log_prob = next_mcmc_state.log_prob


        # resample (only llorente2023 puts it after rejuvinate)
        if resampling.resample_time == ResampleTime.AfterMove:
            particles, ta_log_prob, log_particle_weights = maybe_resample(particles, ta_log_prob, log_particle_weights, log_ess, resample_key)

        ta_log_likelihood = _vmap(slp.log_likelihood)(particles, step_data.data_annealing) if reweighting_type == ReweightingType.Bootstrap else None

        return SMCState(smc_state.iteration+1, particles, log_particle_weights, ta_log_likelihood, ta_log_prob, next_mcmc_state.infos), jax.lax.exp(log_ess)

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
                 rejuvination_attempts: int = 1,
                 *,
                 pconfig: ParallelisationConfig,
                 collect_inference_info: bool = False,
                 show_progress: bool = False,
                 shared_progressbar: tqdm | None = None) -> None:
        
        self.slp = slp
        self.n_particles = n_particles
        self.show_progress = show_progress
        self.progressbar_mngr = ProgressbarManager(
            "SMC for "+self.slp.formatted(),
            shared_progressbar,
            thread_locked=pconfig.vectorisation==VectorisationType.PMAP
        )
        self.pconfig = pconfig
        
        assert pconfig.vectorisation in (VectorisationType.LocalVMAP, VectorisationType.LocalSMAP)
        if pconfig.vectorisation == VectorisationType.LocalSMAP:
            self.vectorisation = "smap"
        else:
            self.vectorisation = "vmap"
        
        rejuvination_kernel, rejuv_init_carry_stat_names = get_mcmc_kernel(slp, rejuvination_regime, collect_inference_info=collect_inference_info, vectorisation=self.vectorisation, return_map=lambda _: None)
        assert len(rejuv_init_carry_stat_names) == 0, "CarryStats in MCMC currently not supported."

        self.rejuvination_regime = rejuvination_regime
        self.rejuvination_kernel: MCMCKernel[None] = rejuvination_kernel
        self.rejuvination_kernel_init_carry_stat_names = rejuv_init_carry_stat_names
        self.rejuvination_attempts = rejuvination_attempts
        self.collect_inference_info = collect_inference_info

        assert tempereture_schedule is not None or data_annealing_schedule is not None
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

        self.smc_step = get_smc_step(slp, reweighting_type, resampling, rejuvination_kernel, self.rejuvination_attempts, self.vectorisation)

        self.cached_smc_scan: Optional[Callable[[SMCState,SMCStepData],Tuple[SMCState,FloatArray]]] = None

    def run(self, rng_key: PRNGKey, particles: StackedTrace, log_prob: Optional[FloatArray] = None):
        # no tempering / annealing weights, we require that this is proper weighting for input particles
        if log_prob is None:
            prior_data_annealing = self.data_annealing_schedule.prior_mask()
            log_prob = jax.vmap(self.slp.log_prob, in_axes=(0,None,None))(particles.data, jnp.array(0.,float), prior_data_annealing)

        # print(log_particle_weights)
        log_particle_weights = jax.numpy.zeros_like(log_prob)
            
        n_particles = particles.n_samples()

        smc_keys = jax.random.split(rng_key, self.n_steps)

        mcmc_infoss = init_inference_infos_for_chains(self.rejuvination_regime, n_particles) if self.collect_inference_info else None

        ta_log_likelihood = jax.numpy.zeros_like(log_prob) if self.reweighting_type == ReweightingType.Bootstrap else None

        init_state = SMCState(jnp.array(0,int), particles.data, log_particle_weights, ta_log_likelihood, log_prob, mcmc_infoss)
        smc_data = SMCStepData(smc_keys, self.tempereture_schedule.temperature, self.data_annealing_schedule.data_annealing)

        self.progressbar_mngr.set_num_samples(self.n_steps)
            
        if self.cached_smc_scan:
            scan_fn = self.cached_smc_scan
        else:
            smc_state_axes = SMCState(iteration=None, particles=0, log_particle_weights=0, ta_log_likelihood=0, ta_log_prob=0, mcmc_infos=0) # type: ignore
            scan_fn = vectorise_scan(self.smc_step, carry_axes=smc_state_axes, pmap_data_axes=1, batch_axis_size=self.n_particles, vectorisation=self.pconfig.vectorisation,
                                     progressbar_mngr=self.progressbar_mngr if self.show_progress else None, get_iternum_fn=lambda carry: carry.iteration)
            self.cached_smc_scan = scan_fn
        
        last_state, ess = parallel_run(scan_fn, (init_state, smc_data), self.n_particles, self.pconfig.vectorisation)

        return last_state, ess 
    



def get_log_Z_ESS(log_weights: FloatArray):
    log_Z = jax.scipy.special.logsumexp(log_weights) - jax.lax.log(float(log_weights.size))
    ESS = jax.lax.exp(jax.scipy.special.logsumexp(log_weights)*2 - jax.scipy.special.logsumexp(log_weights*2))
    return log_Z, ESS

def get_Z_ESS(log_weights: FloatArray):
    log_Z, ESS = get_log_Z_ESS(log_weights)
    return jax.lax.exp(log_Z), ESS

def normalise_log_weights(log_weights: FloatArray):
    return log_weights - jax.scipy.special.logsumexp(log_weights)