import sys
sys.path.insert(0, ".")

from data import *
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from dccxjax import *
import dccxjax.distributions as dist
import numpyro.distributions as numpyro_dist
from kernels import *
from dataclasses import fields
from tqdm.auto import tqdm
from dccxjax.core.branching_tracer import retrace_branching
from time import time

# set_platform("cpu")

import logging
setup_logging(logging.WARN)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)


xs, xs_val, ys, ys_val = get_data_autogp()

# plt.scatter(xs, ys)
# plt.scatter(xs_val, ys_val)
# plt.show()

NODE_TYPES: List[type[GPKernel]] = [Constant, Linear, SquaredExponential, GammaExponential, Periodic, Plus, Times]
NODE_TYPE_PROBS = jnp.array([0.0, 0.21428571428571427, 0.0, 0.21428571428571427, 0.21428571428571427, 0.17857142857142858, 0.17857142857142858])

def covariance_prior(idx: int) -> GPKernel:
    node_type = sample(f"{idx}_node_type", dist.Categorical(NODE_TYPE_PROBS))
    NodeType = NODE_TYPES[node_type]
    if node_type < 5:
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
    if node_type < 5:
        NodeType = NODE_TYPES[node_type]
        params = []
        for field in fields(NodeType):
            field_name = field.name
            log_param = trace[f"{idx}_{field_name}"]
            param = transform_param(field_name, log_param)
            params.append(param)
        return NodeType(*params)
    else:
        NodeType = [Plus, Times][node_type - 5]
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
    if node_type < 5:
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


class SMCDCCConfig(SMCDCC[T]):

    # def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
    #     _active_slps: List[SLP] = []
    #     super().initialise_active_slps(_active_slps, inactive_slps, rng_key)
    #     _active_slps.sort(key=self.model.slp_sort_key)
    #     kernel_key: Set[str] = set()
    #     for slp in _active_slps:
    #         k = get_gp_kernel(slp.decision_representative)
    #         k_key = k.key()
    #         if k.depth() < 5 and k_key not in kernel_key:
    #             active_slps.append(slp)
    #             kernel_key.add(k_key) # deduplicate using commutativity of + and * kernel
    #             log_prob_trace = self.model.log_prob_trace(slp.decision_representative)
    #             log_path_prior: FloatArray = sum((log_prob for addr, (log_prob, _) in log_prob_trace.items() if SuffixSelector("node_type").contains(addr)), start=jnp.array(0,float))
    #             tqdm.write(f"Activate {k_key} with log_path_prior={log_path_prior.item():.4f}")
    #     active_slps.sort(key=self.model.slp_sort_key)
    #     print(f"{len(active_slps)=}")


    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        for node_type in range(len(NODE_TYPE_PROBS) - 2):
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
        
        # selector = PredicateSelector(lambda addr: not addr.endswith("node_type"))
        # regime = MCMCSteps(
        #     MCMCStep(selector, RW(lambda _: dist.Normal(0.,1.), elementwise=True)),
        #     MCMCStep(selector, HMC(10, 0.02))
        # )

        selector = PredicateSelector(lambda addr: not addr == "noise" and not addr.endswith("node_type"))
        regime = MCMCSteps(
            MCMCStep(selector, RW(lambda _: dist.Normal(0.,1.), elementwise=True)),
            MCMCStep(selector, HMC(10, 0.02)),
            MCMCStep(SingleVariable("noise"), HMC(10, 0.02))
        )

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

smc_dcc_obj = SMCDCCConfig(m, verbose=2,
    smc_rejuvination_attempts=8,
    smc_n_particles=10,
    smc_collect_inference_info=True,
    max_iterations = 5,
    n_lmh_update_samples = 1000,
    max_active_slps = 3,
    max_new_active_slps = 3,
    one_inference_run_per_slp = True,
)


# key_sets = [set() for depth in range(10)]
# rng_key = jax.random.PRNGKey(0)
# for _ in tqdm(range(10_000)):
#     rng_key, sample_key = jax.random.split(rng_key)
#     trace, _ = m.generate(sample_key)
#     # trace = equivalence_map(trace)
#     k = get_gp_kernel(trace, ordered=True)
#     d = k.depth()
#     if d < len(key_sets):
#         key_sets[d].add(k.key())

# for depth in range(len(key_sets)):
#     print(f"Depth {depth}: {len(key_sets[depth])}")

# print(key_sets[1])
# print(key_sets[2])
# exit()


t0 = time()
result = smc_dcc_obj.run(jax.random.PRNGKey(0))
result.pprint()
t1 = time()

print(f"Total time: {t1-t0:.3f}s")
comp_time = compilation_time_tracker.get_total_compilation_time_secs()
print(f"Total compilation time: {comp_time:.3f}s ({comp_time / (t1 - t0) * 100:.2f}%)")

slp_weights = list(result.get_slp_weights().items())
slp_weights.sort(key=lambda v: v[1])


map_slp, _ = slp_weights[-1]

weighted_samples = result.get_samples_for_slp(map_slp).unstack()
_, weights = weighted_samples.get()

map_trace, _ = weighted_samples.get_selection(weights.argmax())
k = get_gp_kernel(map_trace)
noise = transform_param("noise", map_trace["noise"]) + 1e-5
xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))
mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)

plt.figure()
plt.scatter(xs, ys)
plt.scatter(xs_val, ys_val)
plt.plot(xs_pred, mvn.numpyro_base.mean, color="black")
q025, q975 = mvnormal_quantile(mvn, 0.025), mvnormal_quantile(mvn, 0.975)
plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
plt.title(map_slp.formatted())
plt.show()



plt.figure()
plt.scatter(xs, ys)
plt.scatter(xs_val, ys_val)
xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))

n_posterior_samples = 1_000
sample_key = jax.random.PRNGKey(0)
slp_weights_array = jnp.array([weight for _, weight in slp_weights])
posterior_over_slps = dist.Categorical(slp_weights_array)

samples = []
for i in tqdm(range(n_posterior_samples), desc="Sample posterior"):
    sample_key, key1, key2, key3 = jax.random.split(sample_key, 4)
    slp_ix = posterior_over_slps.sample(key1)
    slp, _ = slp_weights[slp_ix]
    weighted_samples = result.get_samples_for_slp(slp).unstack()
    _, weights = weighted_samples.get()
    trace_ix = dist.Categorical(weights).sample(key2)
    trace, weight = weighted_samples.get_selection(trace_ix)
    
    k = get_gp_kernel(trace)
    noise = transform_param("noise", trace["noise"]) + 1e-5
    mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)


    if i < 10:
        tqdm.write(f"sample from posterior: {slp.formatted()} noise={noise} with log_prob {m.log_prob(trace)}")

        plt.plot(xs_pred, mvn.numpyro_base.mean, color="black", alpha=0.1)
        q025, q975 = mvnormal_quantile(mvn, 0.025), mvnormal_quantile(mvn, 0.975)
        plt.fill_between(xs_pred, q025, q975, alpha=0.1, color="tab:blue")

    samples.append(mvn.sample(key3))
# plt.show()


samples = jnp.vstack(samples)
m = jnp.mean(samples, axis=0)
q025 = jnp.quantile(samples, 0.025, axis=0)
q975 = jnp.quantile(samples, 0.975, axis=0)

plt.figure()
plt.scatter(xs, ys)
plt.scatter(xs_val, ys_val)
plt.plot(xs_pred, m, color="black")
plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
plt.show()