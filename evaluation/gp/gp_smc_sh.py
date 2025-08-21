from typing import List
import jax
from dccxjax.core import *
from dccxjax.infer import SMCDCC, T, MCMCRegime, MCMCStep, MCMCSteps, RW, HMC, PredicateSelector, SuffixSelector, InferenceResult, LogWeightEstimate, LogWeightEstimateFromSMC
from dccxjax.infer import DataAnnealingSchedule, data_annealing_schedule_from_range, TemperetureSchedule, tempering_schedule_from_sigmoid

from gp import *

from successive_halving import *

class SMCDCCConfig2(SMCDCC[T]):
    def __init__(self, model: Model, *ignore, verbose=0, **config_kwargs) -> None:
        super().__init__(model, *ignore, verbose=verbose, **config_kwargs)
        self.successive_halving: SuccessiveHalving = self.config["successive_halving"]
        self.round_n_particles_to_multiple: int = self.config.get("round_n_particles_to_multiple", 1)
        
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        super().initialise_active_slps(active_slps, inactive_slps, rng_key)
        self.n_phases = self.successive_halving.calculate_num_phases(len(active_slps))
        self.smc_n_particles = self.successive_halving.calculate_num_optimization_steps(len(active_slps))
        if self.round_n_particles_to_multiple > 1:
            self.smc_n_particles = (self.smc_n_particles // self.round_n_particles_to_multiple) * self.round_n_particles_to_multiple
        tqdm.write(f"{len(active_slps)=} {self.n_phases=} {self.smc_n_particles=}")    

    def produce_samples_from_path_prior(self, slp: SLP, rng_key: PRNGKey) -> Tuple[StackedTrace, Optional[FloatArray]]:
        Y: Trace = {addr: value  for addr,value in slp.decision_representative.items() if SuffixSelector("node_type").contains(addr)}
        particles, _ = jax.vmap(slp.generate, in_axes=(0,None))(jax.random.split(rng_key, self.smc_n_particles), Y)
        return StackedTrace(particles, self.smc_n_particles), None
    
    def estimate_path_log_prob(self, slp: SLP, rng_key: PRNGKey) -> FloatArray:
        log_prob_trace = self.model.log_prob_trace(slp.decision_representative)
        log_path_prob = sum((log_prob for addr, (log_prob, _) in log_prob_trace.items() if SuffixSelector("node_type").contains(addr)), start=jnp.array(0,float))
        n_non_leaf_nodes = len([addr for addr, val in slp.decision_representative.items() if addr.endswith("node_type") and val >= len(NODE_CONFIG.NODE_TYPE_PROBS)-2])
        return log_path_prob - n_non_leaf_nodes*jnp.log(2.) # account for equivalence classes (commutativity of bin-op kernels)
    
    def get_SMC_rejuvination_kernel(self, slp: SLP) -> MCMCRegime:        
        selector = PredicateSelector(lambda addr: not addr.endswith("node_type"))
        regime = MCMCSteps(
            MCMCStep(selector, RW(lambda _: dist.Normal(0.,1.), elementwise=True)),
            MCMCStep(selector, HMC(10, 0.02))
        )
        return regime
    
    def get_SMC_data_annealing_schedule(self, slp: SLP) -> Optional[DataAnnealingSchedule]:
        step = round(len(ys)*0.1)
        return data_annealing_schedule_from_range({"obs": range(step,len(ys),step)})
    
    # def get_SMC_tempering_schedule(self, slp: SLP) -> Optional[TemperetureSchedule]:
    #     schedule = tempering_schedule_from_sigmoid(jnp.linspace(-5,5,10))
    #     return schedule
    
    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        inactive_slps.clear()
        inactive_slps.extend(active_slps)
        active_slps.clear()
        
        if self.iteration_counter == self.n_phases:
            return
        
        slp_to_log_weight: List[Tuple[SLP, float]] = []
        for slp in inactive_slps:
            latest_estimate = log_weight_estimates[slp][-1]
            assert isinstance(latest_estimate, LogWeightEstimateFromSMC)
            slp_to_log_weight.append((slp, latest_estimate.get_estimate().item()))
        selected_slps = self.successive_halving.select_active_slps(slp_to_log_weight)
        for slp, log_weight in selected_slps:
            tqdm.write(f"Keep {slp.formatted()} with {log_weight}")
            active_slps.append(slp)
        self.smc_n_particles = self.successive_halving.calculate_num_optimization_steps(len(active_slps))
        if self.round_n_particles_to_multiple > 1:
            self.smc_n_particles = (self.smc_n_particles // self.round_n_particles_to_multiple) * self.round_n_particles_to_multiple
        tqdm.write(f"update active slps {len(active_slps)=} {self.smc_n_particles=}")        
        
    
    