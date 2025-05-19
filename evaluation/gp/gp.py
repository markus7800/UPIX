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
from dccxjax.infer.dcc2 import *
from dccxjax.core.branching_tracer import retrace_branching

import logging
setup_logging(logging.WARN)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)


xs, xs_val, ys, ys_val = get_data_autogp()

# plt.scatter(xs, ys)
# plt.scatter(xs_val, ys_val)
# plt.show()

def covariance_prior(idx: int) -> GPKernel:
    node_type = sample(f"{idx}_node_type", dist.Categorical(jnp.array([0.0, 0.21428571428571427, 0.0, 0.21428571428571427, 0.21428571428571427, 0.17857142857142858, 0.17857142857142858])))
    if node_type < 5:
        NodeType = [Constant, Linear, SquaredExponential, GammaExponential, Periodic][node_type]
        params = []
        for field in fields(NodeType):
            field_name = field.name
            log_param = sample(f"{idx}_{field_name}", dist.Normal(0., 1.))
            param = transform_param(field_name, log_param)
            params.append(param)
        return NodeType(*params)
    else:
        NodeType = [Plus, Times][node_type - 5]
        left = covariance_prior(2*idx)
        right = covariance_prior(2*idx+1)
        return NodeType(left, right)
    

@model
def gaussian_process(xs: jax.Array, ts: jax.Array):
    kernel = covariance_prior(1)
    noise = sample("noise", dist.Normal(0.,1.))
    noise = transform_param("noise", noise) + 1e-5
    cov_matrix = kernel.eval_cov_vec(xs) + noise * jnp.eye(xs.size)
    # MultivariateNormal does cholesky internally
    sample("obs", dist.MultivariateNormal(jnp.zeros_like(xs), covariance_matrix=cov_matrix), observed=ts)
    # llik = jax.scipy.stats.multivariate_normal.logpdf(ts, jnp.zeros_like(xs), cov_matrix)
    # logfactor(llik)
    # sample("obs", dist.Normal(jnp.mean(cov_matrix), noise), observed=ts)

def _get_gp_kernel(trace: Trace, idx: int, ordered: bool) -> GPKernel:
    node_type = trace[f"{idx}_node_type"]
    if node_type < 5:
        NodeType = [Constant, Linear, SquaredExponential, GammaExponential, Periodic][node_type]
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


m = gaussian_process(xs, ys)
m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
m.set_slp_equivalence_class_id_gen(lambda X: get_gp_kernel(X, ordered=True).key(), lambda X: get_gp_kernel(X, ordered=False).key())

class DCCConfig(MCMCDCC[DCC_COLLECT_TYPE]):
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        # HMC L=10 eps=0.02
        regime = MCMCStep(PredicateSelector(lambda addr: not addr.endswith("node_type")), HMC(10, 0.02))
        pprint_mcmc_regime(regime, slp)
        return regime
    # def run_inference(self, slp: SLP, rng_key: PRNGKey) -> InferenceResult:
    #     with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    #         result = super().run_inference(slp, rng_key)
    #         assert isinstance(result, MCMCInferenceResult)
    #         result.last_state.iteration.block_until_ready()
    #     exit()
    #     return result
    
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        _active_slps: List[SLP] = []
        super().initialise_active_slps(_active_slps, inactive_slps, rng_key)
        # we assume that we know that with increasing steps eventually we have unlikely SLPs
        _active_slps.sort(key=self.model.slp_sort_key)
        kernel_key: Set[str] = set()
        for slp in _active_slps:
            k = get_gp_kernel(slp.decision_representative)
            k_key = k.key()
            if k.depth() < 2 and k_key not in kernel_key:
                active_slps.append(slp)
                kernel_key.add(k_key) # deduplicate using commutativity of + and * kernel
                log_prob_trace = self.model.log_prob_trace(slp.decision_representative)
                log_path_prior: FloatArray = sum((log_prob for addr, (log_prob, _) in log_prob_trace.items() if SuffixSelector("node_type").contains(addr)), start=jnp.array(0,float))
                tqdm.write(f"Activate {k_key} with log_path_prior={log_path_prior.item():.4f}")
        active_slps.sort(key=self.model.slp_sort_key)
        print(f"{len(active_slps)=}")


dcc_obj = DCCConfig(m, verbose=2,
              init_n_samples=500,
              init_estimate_weight_n_samples=1_000_000,
              mcmc_n_chains=1,
              mcmc_n_samples_per_chain=600,
              mcmc_collect_for_all_traces=True,
              estimate_weight_n_samples=100,
              max_iterations = 1
            #   return_map=lambda trace: {"start": trace["start"]}
)


t0 = time()
result = dcc_obj.run(jax.random.PRNGKey(0))
result.pprint()

t1 = time()

print(f"Total time: {t1-t0:.3f}s")
comp_time = compilation_time_tracker.get_total_compilation_time_secs()
print(f"Total compilation time: {comp_time:.3f}s ({comp_time / (t1 - t0) * 100:.2f}%)")