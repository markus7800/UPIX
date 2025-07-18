import sys
if len(sys.argv) > 1:
    if sys.argv[1].endswith("cpu"):
        print("Force run on CPU.")
        from dccxjax.backend import *
        set_platform("cpu")

from data import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from dccxjax.all import *
import dccxjax.distributions as dist
from kernels import *
from dataclasses import fields
from tqdm.auto import tqdm
from typing import Tuple, Optional, Dict

import logging
setup_logging(logging.WARN)

xs, xs_val, ys, ys_val = get_data_autogp()

# plt.scatter(xs, ys)
# plt.scatter(xs_val, ys_val)
# plt.show()

def normalise(a: jax.Array): return a / a.sum()

# AutoGP
# N_LEAF_NODE_TYPES = 5
# NODE_TYPES: List[type[GPKernel]] = [Constant, Linear, SquaredExponential, GammaExponential, Periodic, Plus, Times]
# NODE_TYPE_PROBS = normalise(jnp.array([0, 6, 0, 6, 6, 5, 5],float))

# Reichelt
N_LEAF_NODE_TYPES = 4
NODE_TYPES: List[type[GPKernel]] = [UnitRationalQuadratic, UnitPolynomialDegreeOne, UnitSquaredExponential, UnitPeriodic, Plus, Times]
NODE_TYPE_PROBS = normalise(jnp.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1],float))

def covariance_prior(idx: int) -> GPKernel:
    node_type = sample(f"{idx}_node_type", dist.Categorical(NODE_TYPE_PROBS))
    NodeType = NODE_TYPES[node_type]
    if node_type < N_LEAF_NODE_TYPES:
        params = []
        for field in fields(NodeType):
            field_name = field.name
            log_param = sample(f"{idx}_{field_name}", dist.Normal(0., 1.))
            param = transform_param(field_name, log_param)
            params.append(param)
        return NodeType(*params)
    else:
        left = covariance_prior(2*idx)
        right = covariance_prior(2*idx+1)
        return NodeType(left, right) # type: ignore
    
@model
def gaussian_process(xs: jax.Array, ts: jax.Array):
    kernel = covariance_prior(1)
    noise = sample("noise", dist.Normal(0.,1.))
    noise = transform_param("noise", noise) + 1e-5
    cov_matrix = kernel.eval_cov_vec(xs) + noise * jnp.eye(xs.size)
    # MultivariateNormal does cholesky internally
    sample("obs", dist.MultivariateNormal(jnp.zeros_like(xs), covariance_matrix=cov_matrix), observed=ts)

def _get_gp_kernel(trace: Trace, idx: int, ordered: bool) -> GPKernel:
    node_type = trace[f"{idx}_node_type"]
    if node_type < N_LEAF_NODE_TYPES:
        NodeType = NODE_TYPES[node_type]
        params = []
        for field in fields(NodeType):
            field_name = field.name
            log_param = trace[f"{idx}_{field_name}"]
            param = transform_param(field_name, log_param)
            params.append(param)
        return NodeType(*params)
    else:
        NodeType = [Plus, Times][node_type - N_LEAF_NODE_TYPES]
        # de-duplicate
        left = _get_gp_kernel(trace, 2*idx, ordered)
        right = _get_gp_kernel(trace, 2*idx+1, ordered)
        if ordered and left.name() > right.name():
            return NodeType(right, left)
        else:
            return NodeType(left, right)
def get_gp_kernel(trace: Trace, ordered: bool = True) -> GPKernel:
    return _get_gp_kernel(trace, 1, ordered)


def _equivalence_trace(old_trace: Trace, old_idx: int, new_trace: Trace, new_idx: int):
    node_type = old_trace[f"{old_idx}_node_type"]
    new_trace[f"{new_idx}_node_type"] = node_type
    if node_type < N_LEAF_NODE_TYPES:
        for field in fields(NODE_TYPES[node_type]):
            field_name = field.name
            new_trace[f"{new_idx}_{field_name}"] = old_trace[f"{old_idx}_{field_name}"]
    else:
        old_left_cls = NODE_TYPES[old_trace[f"{2*old_idx}_node_type"]]
        old_right_cls = NODE_TYPES[old_trace[f"{2*old_idx+1}_node_type"]]
        if old_left_cls.name() > old_right_cls.name():
            _equivalence_trace(old_trace, 2*old_idx+1, new_trace, 2*new_idx)
            _equivalence_trace(old_trace, 2*old_idx, new_trace, 2*new_idx+1)
        else:
            _equivalence_trace(old_trace, 2*old_idx, new_trace, 2*new_idx)
            _equivalence_trace(old_trace, 2*old_idx+1, new_trace, 2*new_idx+1)
            
def equivalence_map(trace: Trace) -> Trace:
    equivalence_class_representative: Trace = dict()
    _equivalence_trace(trace, 1, equivalence_class_representative, 1)
    if "noise" in trace:
        equivalence_class_representative["noise"] = trace["noise"]
    return equivalence_class_representative


m = gaussian_process(xs, ys)
m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
# m.set_slp_equivalence_class_id_gen(lambda X: get_gp_kernel(X, ordered=True).key(), lambda X: get_gp_kernel(X, ordered=False).key())
m.set_equivalence_map(equivalence_map)

# X, _ = m.generate(jax.random.PRNGKey(2))
# print(get_gp_kernel(X, False))
# print(get_gp_kernel(X, True))
# print(get_gp_kernel(equivalence_map(X), False))

# equivalence_classes: Dict[str, Set[str]] = dict()
# rng_key = jax.random.PRNGKey(0)
# for _ in tqdm(range(10_000)):
#     rng_key, sample_key = jax.random.split(rng_key)
#     trace, _ = m.generate(sample_key)
#     equ_trace = equivalence_map(trace)
#     k = get_gp_kernel(trace, ordered=False)
#     equ_k = get_gp_kernel(equ_trace, ordered=False)
#     d = k.n_internal()
#     if d < 3:
#         equ_key = equ_k.key()
#         if equ_key not in equivalence_classes:
#             equivalence_classes[equ_key] = set([equ_key])
#         equivalence_classes[equ_key].add(k.key())
    

# for k, vals  in equivalence_classes.items():
#     print(k, vals)

# exit()



class SMCDCCConfig(SMCDCC[T]):
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        for node_type in range(N_LEAF_NODE_TYPES):
            if jax.lax.exp(dist.Categorical(NODE_TYPE_PROBS).log_prob(node_type)) > 0:
                rng_key, generate_key = jax.random.split(rng_key)
                trace, _ = self.model.generate(generate_key, {"1_node_type": jnp.array(node_type,int)})
                slp = slp_from_decision_representative(self.model, trace)
                active_slps.append(slp)
                tqdm.write(f"Make SLP {slp.formatted()} active.")

    def produce_samples_from_prior(self, slp: SLP, rng_key: PRNGKey) -> Tuple[StackedTrace, Optional[FloatArray]]:
        Y: Trace = {addr: value  for addr,value in slp.decision_representative.items() if SuffixSelector("node_type").contains(addr)}
        particles, _ = jax.vmap(slp.generate, in_axes=(0,None))(jax.random.split(rng_key, self.smc_n_particles), Y)
        return StackedTrace(particles, self.smc_n_particles), None
    
    def estimate_path_log_prob(self, slp: SLP, rng_key: PRNGKey) -> FloatArray:
        log_prob_trace = self.model.log_prob_trace(slp.decision_representative)
        log_path_prob = sum((log_prob for addr, (log_prob, _) in log_prob_trace.items() if SuffixSelector("node_type").contains(addr)), start=jnp.array(0,float))
        n_non_leaf_nodes = len([addr for addr, val in slp.decision_representative.items() if addr.endswith("node_type") and val >= len(NODE_TYPE_PROBS)-2])
        return log_path_prob - n_non_leaf_nodes*jnp.log(2.) # account for equivalence classes (commutativity of bin-op kernels)
    
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
    
    def run_inference(self, slp: SLP, rng_key: PRNGKey) -> InferenceResult:
        print(f"Run inference with key {rng_key}")
        return super().run_inference(slp, rng_key)
    
    def get_SMC_data_annealing_schedule(self, slp: SLP) -> Optional[DataAnnealingSchedule]:
        step = round(len(ys)*0.1)
        return data_annealing_schedule_from_range({"obs": range(step,len(ys),step)})
    
    # def get_SMC_tempering_schedule(self, slp: SLP) -> Optional[TemperetureSchedule]:
    #     schedule = tempering_schedule_from_sigmoid(jnp.linspace(-5,5,10))
    #     return schedule

# smc_dcc_obj = SMCDCCConfig(m, verbose=2,
#     smc_rejuvination_attempts=8,
#     smc_n_particles=10,
#     smc_collect_inference_info=True,
#     max_iterations = 5,
#     n_lmh_update_samples = 250,
#     max_active_slps = 3,
#     max_new_active_slps = 3,
#     one_inference_run_per_slp = True,
# )

# do_smc = True
# if do_smc:
#     result = timed(smc_dcc_obj.run)(jax.random.PRNGKey(0))
#     result.pprint()

#     slp_weights = list(result.get_slp_weights().items())
#     slp_weights.sort(key=lambda v: v[1])

#     map_slp, _ = slp_weights[-1]

#     weighted_samples = result.get_samples_for_slp(map_slp).unstack()
#     _, weights = weighted_samples.get()

#     map_trace, _ = weighted_samples.get_selection(weights.argmax())
#     k = get_gp_kernel(map_trace)
#     noise = transform_param("noise", map_trace["noise"]) + 1e-5
#     xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))
#     mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)

#     plt.figure()
#     plt.scatter(xs, ys)
#     plt.scatter(xs_val, ys_val)
#     plt.plot(xs_pred, mvn.numpyro_base.mean, color="black")
#     q025, q975 = mvnormal_quantile(mvn, 0.025), mvnormal_quantile(mvn, 0.975)
#     plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
#     plt.title(map_slp.formatted())
#     plt.show()



#     plt.figure()
#     plt.scatter(xs, ys)
#     plt.scatter(xs_val, ys_val)
#     xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))

#     n_posterior_samples = 1_000
#     sample_key = jax.random.PRNGKey(0)
#     slp_weights_array = jnp.array([weight for _, weight in slp_weights])
#     posterior_over_slps = dist.Categorical(slp_weights_array)

#     samples = []
#     for i in tqdm(range(n_posterior_samples), desc="Sample posterior"):
#         sample_key, key1, key2, key3 = jax.random.split(sample_key, 4)
#         slp_ix = posterior_over_slps.sample(key1)
#         slp, _ = slp_weights[slp_ix]
#         weighted_samples = result.get_samples_for_slp(slp).unstack()
#         _, weights = weighted_samples.get()
#         trace_ix = dist.Categorical(weights).sample(key2)
#         trace, weight = weighted_samples.get_selection(trace_ix)
        
#         k = get_gp_kernel(trace)
#         noise = transform_param("noise", trace["noise"]) + 1e-5
#         mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)


#         if i < 10:
#             tqdm.write(f"sample from posterior: {slp.formatted()} noise={noise} with log_prob {m.log_prob(trace)}")

#             plt.plot(xs_pred, mvn.numpyro_base.mean, color="black", alpha=0.1)
#             q025, q975 = mvnormal_quantile(mvn, 0.025), mvnormal_quantile(mvn, 0.975)
#             plt.fill_between(xs_pred, q025, q975, alpha=0.1, color="tab:blue")

#         samples.append(mvn.sample(key3))
#     # plt.show()


#     samples = jnp.vstack(samples)
#     q050 = jnp.median(samples, axis=0)
#     q025 = jnp.quantile(samples, 0.025, axis=0)
#     q975 = jnp.quantile(samples, 0.975, axis=0)

#     plt.figure()
#     plt.scatter(xs, ys)
#     plt.scatter(xs_val, ys_val)
#     plt.plot(xs_pred, q050, color="black")
#     plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
#     plt.show()


import math
import heapq
class SuccessiveHalving:
    def __init__(self, num_total_iterations: int, num_final_arms: int):
        self.num_total_iterations = num_total_iterations
        self.num_final_arms = num_final_arms

    def calculate_num_phases(self, num_arms: int) -> int:
        self.num_phases = 1
        num_active_arms = num_arms
        while num_active_arms > self.num_final_arms:
            num_active_arms = max(math.ceil(num_active_arms / 2), self.num_final_arms)
            self.num_phases += 1

        return self.num_phases

    def calculate_num_optimization_steps(self, num_active_arms: int) -> int:
        return math.floor(self.num_total_iterations / (self.num_phases * num_active_arms))

    def select_active_slps(self, arm2reward: List[Tuple[SLP,float]]) -> List[Tuple[SLP,float]]:
        num_active_arms = len(arm2reward)
        num_to_keep = max(math.ceil(num_active_arms / 2), self.num_final_arms)
        arms_to_keep = heapq.nlargest(num_to_keep, arm2reward, key=lambda v: v[1])
        return arms_to_keep


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
        
        

from dccxjax.infer.variational_inference.optimizers import Adagrad, SGD, Adam
vi_dcc_obj = VIConfig(m, verbose=2,
    init_n_samples=1_000, # 1_000
    advi_n_iter=1_000, # set by successive halving
    advi_L=1, # 1
    advi_optimizer=Adam(0.005), # Adam(0.005)
    elbo_estimate_n_samples=100, # 100
    successive_halving=SuccessiveHalving(1_000_000, 10),
    parallelisation = ParallelisationConfig(
        type=ParallelisationType.MultiProcessingCPU,
        num_workers=32,
        cpu_affinity=True
    )
)

do_vi = True
if do_vi:
    result = timed(vi_dcc_obj.run)(jax.random.PRNGKey(0))
    result.pprint()
    exit()

    slp_weights = list(result.get_slp_weights().items())
    slp_weights.sort(key=lambda v: v[1])

    xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))
    for i in range(min(len(slp_weights),5)):
        slp, weight = slp_weights[-(i+1)]
        print(slp.formatted(), weight)
        g = result.slp_guides[slp]
        
        n = 100
        
        key = jax.random.PRNGKey(0)
        posterior = Traces(g.sample(key, (n,)), n)
        
        samples = []
        for i in range(n):
            key, sample_key = jax.random.split(key)
            trace = posterior.get_ix(i)
            k = get_gp_kernel(trace)
            noise = transform_param("noise", trace["noise"]) + 1e-5
            mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)
            samples.append(mvn.sample(sample_key))

        samples = jnp.vstack(samples)
        m = jnp.mean(samples, axis=0)
        q025 = jnp.quantile(samples, 0.025, axis=0)
        q975 = jnp.quantile(samples, 0.975, axis=0)

        plt.figure()
        plt.title(slp.formatted())
        plt.scatter(xs, ys)
        plt.scatter(xs_val, ys_val)
        plt.plot(xs_pred, m, color="black")
        plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
plt.show()
    


