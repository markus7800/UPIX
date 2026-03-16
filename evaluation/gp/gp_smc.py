from typing import List, Generic
import jax
from upix.core import *
from upix.infer import SMCDCC, T, MCMCRegime, MCMCStep, MCMCSteps, RW, HMC, PredicateSelector, SuffixSelector
from upix.infer import DataAnnealingSchedule, data_annealing_schedule_from_range, TemperetureSchedule, tempering_schedule_from_sigmoid
from upix.parallelisation import parallel_map
from upix.infer.dcc.abstract_dcc import InferenceResult, LogWeightEstimate, AbstractDCC, BaseDCCResult, initialise_active_slps_from_prior
from upix.infer.dcc.mc_dcc import MCInferenceResult, LogWeightedSample
from functools import reduce
from gp import *


from typing import cast

def tree_proposal_cov(resample_node_idx: int, resample_params: Tuple, rng_key: PRNGKey,
                      trace_current: Trace, idx_current: int, 
                      trace_propsed: Trace, idx_proposed: int) -> GPKernel:
    if idx_current == resample_node_idx:
        node_type = trace_current[f"{idx_current}_node_type"]
        if node_type < NODE_CONFIG.N_LEAF_NODE_TYPES:
            # split
            (split_nodetype, leaf_nodetype) = cast(Tuple[jax.Array,jax.Array], resample_params)
            trace_propsed[f"{idx_proposed}_node_type"] = split_nodetype
            
            # write current node to left
            trace_propsed[f"{2*idx_proposed}_node_type"] = node_type
            left_params = []
            for field in fields(NODE_CONFIG.NODE_TYPES[node_type]):
                rng_key, param_key = jax.random.split(rng_key)
                field_name = field.name
                log_param = trace_current[f"{idx_current}_{field_name}"]  + jax.random.normal(param_key) * 0.1 # rw
                trace_propsed[f"{2*idx_proposed}_{field_name}"] = log_param
                param = transform_param(field_name, log_param)
                left_params.append(transform_param(field_name, param))
            
            # create new node for right
            trace_propsed[f"{2*idx_proposed+1}_node_type"] = leaf_nodetype
            right_params = []
            for field in fields(NODE_CONFIG.NODE_TYPES[leaf_nodetype]):
                rng_key, param_key = jax.random.split(rng_key)
                field_name = field.name
                log_param = jax.random.normal(param_key) # prior
                trace_propsed[f"{2*idx_proposed+1}_{field_name}"] = log_param
                param = transform_param(field_name, log_param)
                right_params.append(transform_param(field_name, param))
                
            SplitNodeType = NODE_CONFIG.NODE_TYPES[split_nodetype]
            LeftNodeType = NODE_CONFIG.NODE_TYPES[node_type]
            RightNodeType = NODE_CONFIG.NODE_TYPES[leaf_nodetype]
            return SplitNodeType(
                LeftNodeType(*left_params),
                RightNodeType(*right_params)
            )
        else:
            # discard
            right, = cast(Tuple[int], resample_params)
            return tree_proposal_cov(resample_node_idx, resample_params, rng_key,
                          trace_current, 2*idx_current+right, trace_propsed, idx_proposed)
    
    else:
        node_type = trace_current[f"{idx_current}_node_type"]
        trace_propsed[f"{idx_proposed}_node_type"] = node_type
        if node_type < NODE_CONFIG.N_LEAF_NODE_TYPES:
            NodeType = NODE_CONFIG.NODE_TYPES[node_type]
            params = []
            for field in fields(NodeType):
                rng_key, param_key = jax.random.split(rng_key)
                field_name = field.name
                log_param = trace_current[f"{idx_current}_{field_name}"] + jax.random.normal(param_key) * 0.1 # rw
                trace_propsed[f"{idx_proposed}_{field_name}"] = log_param
                param = transform_param(field_name, log_param)
                params.append(param)
            return NodeType(*params)
        else:
            NodeType = [Plus, Times][node_type - NODE_CONFIG.N_LEAF_NODE_TYPES]
            left_key, right_key = jax.random.split(rng_key)
            left = tree_proposal_cov(resample_node_idx, resample_params, left_key,
                                     trace_current, 2*idx_current, trace_propsed, 2*idx_proposed)
            right = tree_proposal_cov(resample_node_idx, resample_params, right_key,
                                      trace_current, 2*idx_current+1, trace_propsed, 2*idx_proposed+1)
            return NodeType(left, right)

from pprint import pprint
def tree_proposal(proposal_key: PRNGKey, resample_node_idx: int, resample_params: Tuple, trace_current: Trace, xs: jax.Array, ts: jax.Array):
    trace_proposed: Trace = dict()
    cov_key, noise_key = jax.random.split(proposal_key)
    kernel = tree_proposal_cov(resample_node_idx, resample_params, cov_key, trace_current, 1, trace_proposed, 1)
    noise = jax.random.normal(noise_key) # prior
    trace_proposed["noise"] = noise
    # pprint(trace_proposed)
    
    noise = transform_param("noise", noise) + 1e-5
    cov_matrix = kernel.eval_cov_vec(xs) + noise * jnp.eye(xs.size)
    ll = dist.MultivariateNormal(jnp.zeros_like(xs), covariance_matrix=cov_matrix).log_prob(ts)
    
    return trace_proposed, ll
    
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
        _generate = parallel_map(slp.generate, in_axes=(0,None), out_axes=0, batch_axis_size=self.smc_n_particles, pconfig=self.pconfig, promote_to_global=True)
        particles, _ = _generate(jax.random.split(rng_key, self.smc_n_particles),Y)
        return StackedTrace(particles, self.smc_n_particles), None
    
    def estimate_path_log_prob(self, slp: SLP, rng_key: PRNGKey) -> FloatArray:
        log_prob_trace = self.model.log_prob_trace(slp.decision_representative)
        log_path_prob = sum((log_prob for addr, (log_prob, _) in log_prob_trace.items() if SuffixSelector("node_type").contains(addr)), start=jnp.array(0,float))
        n_non_leaf_nodes = len([addr for addr, val in slp.decision_representative.items() if addr.endswith("node_type") and val >= len(NODE_CONFIG.NODE_TYPE_PROBS)-2])
        return log_path_prob# - n_non_leaf_nodes*jnp.log(2.) # account for equivalence classes (commutativity of bin-op kernels)
    
    def get_SMC_rejuvination_kernel(self, slp: SLP) -> MCMCRegime:
        selector = PredicateSelector(lambda addr: not addr.endswith("node_type"))
        regime = MCMCSteps(
            MCMCStep(selector, RW(lambda _: dist.Normal(0.,1.), elementwise=True)),
            MCMCStep(selector, HMC(10, 0.02))
        )
        return regime
    
    def get_SMC_data_annealing_schedule(self, slp: SLP) -> Optional[DataAnnealingSchedule]:
        n_data = self.config["n_data"]
        step = round(n_data*0.1)
        return data_annealing_schedule_from_range({"obs": range(step,n_data,step)})
    
    # def get_SMC_tempering_schedule(self, slp: SLP) -> Optional[TemperetureSchedule]:
    #     schedule = tempering_schedule_from_sigmoid(jnp.linspace(-5,5,10))
    #     return schedule
    

    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        inactive_slps.extend(active_slps)
        active_slps.clear()

        if self.iteration_counter == self.max_iterations:
            return

        combined_inference_results: Dict[SLP, InferenceResult] = {slp: reduce(lambda x, y: x.combine_results(y), results) for slp, results in inference_results.items()}
        combined_log_weight_estimates: Dict[SLP, LogWeightEstimate] = {slp: reduce(lambda x, y: x.combine_estimates(y), results) for slp, results in log_weight_estimates.items()}

        slp_log_weights = self.compute_slp_log_weight(combined_log_weight_estimates)
        slps: List[SLP] = []
        log_weight_list: List[FloatArray] = []
        for slp, log_weight in slp_log_weights.items():
            slps.append(slp)
            log_weight_list.append(log_weight)
        log_weights = jnp.array(log_weight_list)
        
        slp_to_proposal_prob: Dict[SLP, FloatArray] = dict()
        n_samples = self.config.get("n_update_samples", 25)
        for _ in tqdm(range(n_samples), desc="Determining new active SLPs", disable=self.disable_progress):
            rng_key, select_slp_key, select_trace_key, proposed_key = jax.random.split(rng_key, 4)
            # select SLP proportional to the logweight
            slp = slps[jax.random.categorical(select_slp_key, log_weights)]
            slp_results = combined_inference_results[slp]
            assert isinstance(slp_results, MCInferenceResult)
            
            # select trace from SLP inference result
            weighted_sample: LogWeightedSample[Trace] = slp_results.get_weighted_sample(lambda x: x)
            trace_ix = jax.random.categorical(select_trace_key, weighted_sample.log_weights.reshape(-1))
            sample_ix, chain_ix = jnp.unravel_index(trace_ix, weighted_sample.log_weights.shape)
            trace: Trace = jax.tree.map(lambda v: v[sample_ix, chain_ix, ...], weighted_sample.values.data)
            
            select_leaf_key, select_move_key, select_split_node, select_leaf_node, select_discard_node, tree_key = jax.random.split(proposed_key, 6)
            leaf_nodes = [(addr, val) for addr, val in trace.items() if addr.endswith("node_type") and val < NODE_CONFIG.N_LEAF_NODE_TYPES]
            leaf = leaf_nodes[jax.random.randint(select_leaf_key, (), 0, len(leaf_nodes))]
            leaf_ix = int(leaf[0][:-len("_node_type")])
            
            move = ["split", "discard"][jax.random.bernoulli(select_move_key, 0.5, ()).item()] if leaf[0] != "1_node_type" else "split"
            
            split_node = jax.random.randint(select_split_node, (), 0, 2) + NODE_CONFIG.N_LEAF_NODE_TYPES
            
            leaf_node = jax.random.randint(select_leaf_node, (), 0, NODE_CONFIG.N_LEAF_NODE_TYPES)
            
            discard_node = int(jax.random.bernoulli(select_discard_node, 0.5))
            
            resample_node_ix = leaf_ix if move == "split" else leaf_ix // 2
            
            # tqdm.write(str(get_gp_kernel(trace)))
            # tqdm.write(f"{resample_node_ix=} {move=} {split_node=} {leaf_node=} {discard_node=}")
            resample_params = (split_node, leaf_node) if move == "split" else (discard_node,)
            
            n_proposals = self.config.get("n_proposals_per_update_sample", 100)
            xs, ts = self.model.args
            # print(get_gp_kernel(tree_proposal(tree_key, resample_node_ix, resample_params, trace, xs, ts)[0]))
                        
            traces_proposed, loglikelihoods = jax.vmap(tree_proposal, in_axes=(0,None,None,None,None,None))(
                jax.random.split(tree_key, n_proposals), resample_node_ix, resample_params, trace, xs, ts)
            traces_proposed = StackedTrace(traces_proposed, n_proposals)
            amax = jnp.argmax(loglikelihoods).item()
            trace_proposed = traces_proposed.get_ix(amax)
            loglikelihood: FloatArray = loglikelihoods[amax]
            # tqdm.write(f" -> {get_gp_kernel(trace_proposed)}")
            
            if self.model.equivalence_map is not None:
                trace_proposed = self.model.equivalence_map(trace_proposed)

            # check if we know slp of proposed trace
            matched_slp = next(filter(lambda _slp: _slp.path_indicator(trace_proposed) != 0, inactive_slps), None)
            if matched_slp is None:
                matched_slp = slp_from_decision_representative(self.model, trace_proposed)
                if self.verbose >= 2:
                    tqdm.write(f"Discovered SLP {matched_slp.formatted()}.")
                inactive_slps.append(matched_slp)

            slp_to_proposal_prob[matched_slp] = jnp.maximum(slp_to_proposal_prob.get(matched_slp, -jnp.inf), loglikelihood)

    
        # pick top with respect to loglikelihood
        slp_to_proposal_prob_list = list(slp_to_proposal_prob.items())
        slp_to_proposal_prob_list.sort(key=lambda v: v[1].item(), reverse=True)

        new_active_slp_count = 0
        for slp, prob in slp_to_proposal_prob_list:
            if len(active_slps) >= self.max_active_slps:
                break

            already_performed_inference = slp in slp_log_weights
            if self.one_inference_run_per_slp and already_performed_inference:
                continue
            if (not already_performed_inference) and (new_active_slp_count >= self.max_new_active_slps):
                continue

            tqdm.write(f"Make SLP {slp.formatted()} active (already performed inference = {already_performed_inference}).")
            active_slps.append(slp)
            inactive_slps.remove(slp)
            new_active_slp_count += (not already_performed_inference)
            