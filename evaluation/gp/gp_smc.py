from typing import List, Generic
import jax
from dccxjax.core import *
from dccxjax.infer import SMCDCC, T, MCMCRegime, MCMCStep, MCMCSteps, RW, HMC, PredicateSelector, SuffixSelector
from dccxjax.infer import DataAnnealingSchedule, data_annealing_schedule_from_range, TemperetureSchedule, tempering_schedule_from_sigmoid

from gp import *

class SMCDCCConfig(SMCDCC[T], Generic[T]):
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        for node_type in range(NODE_CONFIG.N_LEAF_NODE_TYPES):
            if jax.lax.exp(dist.Categorical(NODE_CONFIG.NODE_TYPE_PROBS).log_prob(node_type)) > 0:
                rng_key, generate_key = jax.random.split(rng_key)
                trace, _ = self.model.generate(generate_key, {"1_node_type": jnp.array(node_type,int)})
                slp = slp_from_decision_representative(self.model, trace)
                active_slps.append(slp)
                tqdm.write(f"Make SLP {slp.formatted()} active.")

    def produce_samples_from_path_prior(self, slp: SLP, rng_key: PRNGKey) -> Tuple[StackedTrace, Optional[FloatArray]]:
        Y: Trace = {addr: value  for addr,value in slp.decision_representative.items() if SuffixSelector("node_type").contains(addr)}
        particles, _ = jax.vmap(slp.generate, in_axes=(0,None))(jax.random.split(rng_key, self.smc_n_particles), Y)
        return StackedTrace(particles, self.smc_n_particles), None
    
    def estimate_path_log_prob(self, slp: SLP, rng_key: PRNGKey) -> FloatArray:
        log_prob_trace = self.model.log_prob_trace(slp.decision_representative)
        log_path_prob = sum((log_prob for addr, (log_prob, _) in log_prob_trace.items() if SuffixSelector("node_type").contains(addr)), start=jnp.array(0,float))
        n_non_leaf_nodes = len([addr for addr, val in slp.decision_representative.items() if addr.endswith("node_type") and val >= len(NODE_CONFIG.NODE_TYPE_PROBS)-2])
        return log_path_prob# - n_non_leaf_nodes*jnp.log(2.) # account for equivalence classes (commutativity of bin-op kernels)
    
    def get_SMC_rejuvination_kernel(self, slp: SLP) -> MCMCRegime:
        # regime = MCMCStep(PredicateSelector(lambda addr: not addr.endswith("node_type")), HMC(10, 0.02))
        
        selector = PredicateSelector(lambda addr: not addr.endswith("node_type"))
        regime = MCMCSteps(
            MCMCStep(selector, RW(lambda _: dist.Normal(0.,1.), elementwise=True)),
            MCMCStep(selector, HMC(10, 0.02))
        )

        # selector = PredicateSelector(lambda addr: not addr == "noise" and not addr.endswith("node_type"))
        # regime = MCMCSteps(
        #     MCMCStep(selector, RW(lambda _: dist.Normal(0.,1.), elementwise=True)),
        #     MCMCStep(selector, HMC(10, 0.02)),
        #     MCMCStep(SingleVariable("noise"), HMC(10, 0.02))
        # )

        # pprint_mcmc_regime(regime, slp)
        return regime
    
    def get_SMC_data_annealing_schedule(self, slp: SLP) -> Optional[DataAnnealingSchedule]:
        step = round(len(ys)*0.1)
        return data_annealing_schedule_from_range({"obs": range(step,len(ys),step)})
    
    # def get_SMC_tempering_schedule(self, slp: SLP) -> Optional[TemperetureSchedule]:
    #     schedule = tempering_schedule_from_sigmoid(jnp.linspace(-5,5,10))
    #     return schedule
    