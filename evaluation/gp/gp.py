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


m = gaussian_process(xs, ys)
m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())

X, lp = m.generate(
    jax.random.PRNGKey(0),
    Y={
        "noise": jnp.array(0.12390123120559722,float),
        "1_node_type": jnp.array(3,int),
        "1_lengthscale": jnp.array(0.942970533446119,float),
        "1_gamma": jnp.array(0.13392275765318448,float),
        "1_amplitude": jnp.array(1.5250689085124804,float)
    }
)
slp = slp_from_decision_representative(m, X)

if False:
    N_samples = 500
    print(f"{N_samples=}")

    do_vmap_step = True
    @jax.jit
    def _is(rng_key: PRNGKey):
        res, _ = retrace_branching(lambda key: m.generate(key, Y={"1_node_type": jnp.array(3,int)}), slp.branching_decisions)(rng_key)
        return res
    keys = jax.random.split(jax.random.PRNGKey(0), N_samples)
    t0 = time()
    # traces, lps = jax.vmap(jax.jit(_is))(keys)
    def _f(_, rng_key):
        return (None, _is(rng_key))
    if do_vmap_step:
        _, (traces, lps) = jax.lax.scan(jax.vmap(_f), None, keys.reshape(N_samples,1,2))
    else:
        # fast
        _, (traces, lps) = jax.lax.scan(_f, None, keys)
    # slow-ish
    lps.block_until_ready()
    t1 = time()
    print(f"Finished IS in {t1-t0:.3f} s")
    # plt.hist(lps,density=True,bins=100)
    # plt.show()

    node_type = traces["1_node_type"]
    del traces["1_node_type"]
    t0 = time()
    # grads = jax.vmap(jax.jit(jax.grad(lambda x: slp.log_prob(x | {"1_node_type": jnp.array(3,int)}))))(traces)
    def _f_grad(acc, trace):
        g = jax.grad(lambda x: slp.log_prob(x | {"1_node_type": jnp.array(3,int)}))(trace)
        return (acc + g["noise"]) / 2, g # force sequential computation
    # _, grads = jax.lax.scan(lambda _, trace: (None, jax.grad(lambda x: slp.log_prob(x | {"1_node_type": jnp.array(3,int)}))(trace)), None, traces)
    if do_vmap_step:
    # slow
        acc, grads = jax.lax.scan(jax.vmap(_f_grad), jnp.array([0],float), traces)#jax.tree.map(lambda v: v.reshape(v.shape[0],1,*v.shape[1:]), traces))
    else:
        # fast
        acc, grads = jax.lax.scan(_f_grad, jnp.array(0,float), traces)
    acc.block_until_ready()
    t1 = time()
    print(acc)
    print(f"Finished Grad in {t1-t0:.3f} s")
    traces["1_node_type"] = node_type


    hmc_kernel = HMC(10, 0.02).make_kernel(
        GibbsModel(slp, PredicateSelector(lambda addr: not addr.endswith("node_type")), {"1_node_type": jnp.array(3,int)}),
        0,
        False
    )
    
    from dccxjax.infer.mcmc import KernelState, CarryStats
    def hmc_step(state: KernelState, rng_key: PRNGKey):
        rng_key, _ = jax.random.split(rng_key)
        new_state = hmc_kernel(rng_key, jnp.array(1.,float), state)
        return new_state, new_state.carry_stats["position"]
    initial_kernel_state = KernelState(CarryStats(position=X, log_prob=lp), None)

    # t0 = time()
    # slow
    # last_state, positions = jax.lax.scan(jax.vmap(hmc_step), broadcast_jaxtree(initial_kernel_state, (1,)), keys.reshape(N_samples,1,2))
    # fast
    # last_state, positions = jax.lax.scan(hmc_step, initial_kernel_state, keys)
    # last_state.carry_stats["log_prob"].block_until_ready()
    # print(last_state.carry_stats["log_prob"])
    # t1 = time()
    # print(f"Finished HMC seq in {t1-t0:.3f} s")
    # plt.plot(positions["noise"])
    # plt.plot(positions["1_amplitude"])
    # plt.plot(positions["1_gamma"])
    # plt.plot(positions["1_lengthscale"])
    # plt.show()

    from dccxjax.infer.mcmc import get_mcmc_kernel
    # problems for both
    # regime = MCMCStep(PredicateSelector(lambda addr: not addr.endswith("node_type")), RW(gaussian_random_walk(0.02))),
    regime = MCMCStep(PredicateSelector(lambda addr: not addr.endswith("node_type")), HMC(10, 0.02))

    # slow
    kernel_step, _ = get_mcmc_kernel(slp, regime, return_map=lambda x: x.position, vectorised=True)
    initial_mcmc_state = MCMCState(jnp.array(0,int), jnp.array(1.,float), broadcast_jaxtree(X, (1,)), broadcast_jaxtree(lp, (1,)), CarryStats(), None)

    
    t0 = time()
    last_state, positions = jax.lax.scan(kernel_step, initial_mcmc_state, keys)
    last_state.log_prob.block_until_ready()
    print(last_state.log_prob)
    t1 = time()
    print(f"Finished HMC seq 1 in {t1-t0:.3f} s")

    # fast
    kernel_step, _ = get_mcmc_kernel(slp, regime, return_map=lambda x: x.position, vectorised=False)
    initial_mcmc_state = MCMCState(jnp.array(0,int), jnp.array(1.,float), X, lp, CarryStats(), None)
    
    t0 = time()
    last_state, positions = jax.lax.scan(kernel_step, initial_mcmc_state, keys)
    last_state.log_prob.block_until_ready()
    print(last_state.log_prob)
    t1 = time()
    print(f"Finished HMC seq 2 in {t1-t0:.3f} s")


    n_chains = 1
    n_samples_per_chain = N_samples
    mcmc_obj = MCMC(
        slp,
        regime,
        n_chains,
        collect_inference_info=True, progress_bar=True, return_map=lambda x: x
    )
    pprint_mcmc_regime(mcmc_obj.regime, slp)
    t0 = time()
    last_state, r = mcmc_obj.run(jax.random.PRNGKey(0), StackedTrace(broadcast_jaxtree(X, (n_chains,)), n_chains), n_samples_per_chain=n_samples_per_chain)
    print(last_state.log_prob)
    t1 = time()
    print(f"Finished HMC MCMC in {t1-t0:.3f} s")
    for info in last_state.infos:
        print(summarise_mcmc_info(info, n_samples_per_chain))

    exit()

    # from pprint import pprint
    # pprint(m.log_prob_trace(X))

@jax.jit
def mycholesky(A: jax.Array):
    def body_fun(i: int, L: jax.Array):
        print(L, i)
        jax.debug.print("i={i}", i=i)
        Lii = jnp.sqrt(A[i,i] - jnp.dot(L[:i,i], L[:i,i]))
        L_ = A[i,i:] - jnp.matmul(L[:i,i], L[:i,i:])
        return L.at[i,i].set(Lii).at[i,i:].set(L_ / Lii)
    return jax.lax.fori_loop(0, A.shape[0], body_fun, jnp.zeros_like(A))

if False:
    k = get_gp_kernel(X)
    print(k.__repr__(True))
    noise = transform_param("noise", X["noise"])
    print(noise)
    cov_matrix = k.eval_cov_vec(xs) + (1e-5+noise)*jnp.eye(xs.shape[0])
    print(cov_matrix)
    L = jnp.linalg.cholesky(cov_matrix, upper=False)
    print("L =")
    print(L)
    L1 = L

    L = mycholesky(cov_matrix)
    print(L)
    print(jnp.isclose(L1, L).all())

    # A = cov_matrix
    # L = jnp.zeros_like(A)
    # for i in range(A.shape[0]):
    #     Lii = jnp.sqrt(A[i,i] - jnp.dot(L[:i,i], L[:i,i]))
    #     L = L.at[i,i].set(Lii)
    #     L_ = A[i,i:] - jnp.matmul(L[:i,i], L[:i,i:])
    #     L = L.at[i,i:].set(L_ / Lii)
    # print("L2 =")
    # print(L)
    # print(jnp.isclose(L1, L).all())

    xs_pred = jnp.concatenate((xs,xs_val))
    pp = k.posterior_predictive(xs, ys, noise, xs_pred, noise)
    ppn = numpyro_dist.Normal(pp.numpyro_base.mean, jnp.sqrt(pp.numpyro_base.variance))
    q975 = ppn.icdf(0.975)
    q025 = ppn.icdf(0.025)

    posterior = pp.sample(jax.random.PRNGKey(0), (100,))

    # plt.fill_between(xs_pred, q025, q975, alpha=0.25)
    # plt.scatter(xs, ys, s=2)
    # plt.scatter(xs_val, ys_val, s=2)
    # plt.plot(xs_pred, q025)
    # plt.plot(xs_pred, q975)
    # for i in range(100):
    #     plt.plot(xs_pred, posterior[i,:], color="gray", alpha=0.05)
    # plt.show()

# exit()


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
            k_key = str(k)
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