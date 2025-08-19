from typing import List
import jax
from dccxjax.core import *
from dccxjax.infer import VIDCC, Guide, PredicateSelector, MeanfieldNormalGuide, InferenceResult, LogWeightEstimateFromADVI, LogWeightEstimate

from gp import *

from successive_halving import *


class VIConfig(VIDCC):
    # def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
    #     for node_type in range(N_LEAF_NODE_TYPES):
    #         if jax.lax.exp(dist.Categorical(NODE_TYPE_PROBS).log_prob(node_type)) > 0:
    #             rng_key, generate_key = jax.random.split(rng_key)
    #             trace, _ = self.model.generate(generate_key, {"1_node_type": jnp.array(node_type,int)})
    #             slp = slp_from_decision_representative(self.model, trace)
    #             active_slps.append(slp)
    #             tqdm.write(f"Make SLP {slp.formatted()} active.")
    
    def __init__(self, model: Model, *ignore, verbose=0, **config_kwargs) -> None:
        super().__init__(model, *ignore, verbose=verbose, **config_kwargs)
        self.successive_halving: SuccessiveHalving = self.config["successive_halving"]
                
    def get_guide(self, slp: SLP) -> Guide:
        selector = PredicateSelector(lambda addr: not addr.endswith("node_type"))
        return MeanfieldNormalGuide(slp, selector, 0.1)
        # return FullRankNormalGuide(slp, selector, 0.1)
    
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        super().initialise_active_slps(active_slps, inactive_slps, rng_key)
        self.n_phases = self.successive_halving.calculate_num_phases(len(active_slps))
        self.advi_n_iter = self.successive_halving.calculate_num_optimization_steps(len(active_slps))
        # self.advi_n_iter = 1000
        # self.n_phases = 1
        tqdm.write(f"{len(active_slps)=} {self.n_phases=} {self.advi_n_iter=}")
    
    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        inactive_slps.clear()
        inactive_slps.extend(active_slps)
        active_slps.clear()
        
        if self.iteration_counter == self.n_phases:
            return
        
        slp_to_log_weight: List[Tuple[SLP, float]] = []
        for slp in inactive_slps:
            latest_estimate = log_weight_estimates[slp][-1]
            assert isinstance(latest_estimate, LogWeightEstimateFromADVI)
            slp_to_log_weight.append((slp, latest_estimate.get_estimate().item()))
        selected_slps = self.successive_halving.select_active_slps(slp_to_log_weight)
        for slp, log_weight in selected_slps:
            tqdm.write(f"Keep {slp.formatted()} with {log_weight}")
            active_slps.append(slp)
        self.advi_n_iter = self.successive_halving.calculate_num_optimization_steps(len(active_slps))
        tqdm.write(f"update active slps {len(active_slps)=} {self.advi_n_iter=}")        
        
    

