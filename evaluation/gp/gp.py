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
def gaussian_process(ts: jax.Array):
    kernel = covariance_prior(1)
    noise = sample("noices", dist.Normal(0.,1.))
    noise = transform_param("noise", noise) + 1e-5
    cov_matrix = kernel.eval_cov_vec(ts) + noise * jnp.eye(ts.size)
    # MultivariateNormal does cholesky internally
    sample("obs", dist.MultivariateNormal(covariance_matrix=cov_matrix), observed=ts)

def _get_gp_kernel(trace: Trace, idx: int) -> GPKernel:
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
        left = _get_gp_kernel(trace, 2*idx)
        right = _get_gp_kernel(trace, 2*idx+1)
        if left.name() < right.name():
            return NodeType(left, right)
        else:
            return NodeType(right, left)
def get_gp_kernel(trace: Trace) -> GPKernel:
    return _get_gp_kernel(trace, 1)


m = gaussian_process(ys)
m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())


class DCCConfig(MCMCDCC[DCC_COLLECT_TYPE]):
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        # HMC L=10 eps=0.02
        raise NotImplementedError
    
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        _active_slps: List[SLP] = []
        super().initialise_active_slps(_active_slps, inactive_slps, rng_key)
        # we assume that we know that with increasing steps eventually we have unlikely SLPs
        _active_slps.sort(key=self.model.slp_sort_key)
        kernel_key: Set[str] = set()
        for slp in _active_slps:
            k = get_gp_kernel(slp.decision_representative)
            k_key = str(k)
            if k.depth() < 4 and k_key not in kernel_key:
                active_slps.append(slp)
                kernel_key.add(k_key) # deduplicate using commutativity of + and * kernel
                log_prob_trace = self.model.log_prob_trace(slp.decision_representative)
                log_path_prior: FloatArray = sum((log_prob for addr, log_prob in log_prob_trace.items() if SuffixSelector("node_type").contains(addr)), start=jnp.array(0,float))
                tqdm.write(f"Activate {k_key} with log_path_prior={log_path_prior.item():.4f}")
        active_slps.sort(key=self.model.slp_sort_key)
        print(f"{len(active_slps)=}")
        exit()


dcc_obj = DCCConfig(m, verbose=2,
              init_n_samples=10_000,
              init_estimate_weight_n_samples=1_000_000,
              mcmc_n_chains=10,
              mcmc_n_samples_per_chain=25_000,
              mcmc_collect_for_all_traces=True,
              estimate_weight_n_samples=10_000_000,
            #   return_map=lambda trace: {"start": trace["start"]}
)


t0 = time()

result = dcc_obj.run(jax.random.PRNGKey(0))
result.pprint()

t1 = time()