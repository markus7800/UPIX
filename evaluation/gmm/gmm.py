import sys
sys.path.insert(0, ".")

from dccxjax import *
import jax
import jax.numpy as jnp
import dccxjax.distributions as dist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List, NamedTuple
from time import time

from dccxjax.infer.mcmc import MCMCState, InferenceInfos, get_inference_regime_mcmc_step_for_slp, add_progress_bar
from dccxjax.infer.dcc import DCC_Result
from dccxjax.core.branching_tracer import retrace_branching

from dccxjax.infer.estimate_Z import estimate_Z_for_SLP_from_sparse_mixture

import logging
setup_logging(logging.DEBUG)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)

t0 = time()

lam = 3
delta = 5.0
xi = 0.0
kappa = 0.01
alpha = 2.0
beta = 10.0

def gmm(ys: jax.Array):

    N = ys.shape[0]
    K = sample("K", dist.Poisson(lam-1)) + 1
    w = sample("w", dist.Dirichlet(jnp.full((K,), delta)))
    mus = sample("mus", dist.Normal(jnp.full((K,), xi), jnp.full((K,), 1/jax.lax.sqrt(kappa))))
    vars = sample("vars", dist.InverseGamma(jnp.full((K,), alpha), jnp.full((K,), beta)))
    zs = sample("zs", dist.Categorical(jax.lax.broadcast(w, (N,))))
    sample("ys", dist.Normal(mus[zs], jax.lax.sqrt(vars[zs])), observed=ys)


ys = jnp.array([
    -7.87951290075215, -23.251364738213493, -5.34679518882793, -3.163770449770572,
    10.524424782864525, 5.911987013277482, -19.228378698266436, 0.3898087330050574,
    8.576922415766697, 7.727416085566447, -18.043123523482492, 9.108136117789305,
    29.398734347901787, 2.8578485031858003, -20.716691460295685, -18.5075008084623,
    -21.52338318392563, 10.062657028986715, -18.900545157827718, 3.339430437507262,
    3.688098690412526, 4.209808727262307, 3.371091291010914, 30.376814419984456,
    12.778653273596902, 28.063124205174137, 10.70527515161964, -18.99693615834304,
    8.135342537554163, 29.720363913218446, 29.426043027354385, 28.40516772785764,
    31.975585225366686, -20.642437143912638, 30.84807631345935, -21.46602061526647,
    12.854676808303978, 30.685416799345685, 5.833520737134923, 7.602680172973942,
    10.045516408942117, 28.62342173081479, -20.120184774438087, -18.80125468061715,
    12.849708921404385, 31.342270731653656, 4.02761078481315, -19.953549865339976,
    -2.574052170014683, -21.551814470820258, -2.8751904316333268,
    13.159719198798443, 8.060416669497197, 12.933573330915458, 0.3325664001681059,
    11.10817217269102, 28.12989207125211, 11.631846911966806, -15.90042467317705,
    -0.8270272159702201, 11.535190070081708, 4.023136673956579,
    -22.589713328053048, 28.378124912868305, -22.57083855780972,
    29.373356677376297, 31.87675796607244, 2.14864533495531, 12.332798078071061,
    8.434664672995181, 30.47732238916884, 11.199950328766784, 11.072188217008367,
    29.536932243938097, 8.128833670186253, -16.33296115562885, 31.103677511944685,
    -20.96644212192335, -20.280485886015406, 30.37107537844197, 10.581901339669418,
    -4.6722903116912375, -20.320978011296315, 9.141857987635252, -18.6727012563551,
    7.067728508554964, 5.664227155828871, 30.751158861494442, -20.198961378110013,
    -4.689645356611053, 30.09552608716476, -19.31787364001907, -22.432589846769154,
    -0.9580412415863696, 14.180597007125487, 4.052110659466889,
    -18.978055134755582, 13.441194891615718, 7.983890038551439, 7.759003567480592
])

# ys = ys[:10]

from dccxjax.core.samplecontext import GenerateCtx, LogprobCtx
with GenerateCtx(jax.random.PRNGKey(0)) as ctx1:
    gmm(ys)
    print(ctx1.log_likelihood + ctx1.log_prior)


# X = {'mus': jnp.array([ 5.0535045,  0.2384372, 22.40249  , -5.599992 ]), 'vars': jnp.array([10.463084 , 46.026665 ,  4.8527174,  3.1191897]), 'w': jnp.array([-0.00192596,  0.37337512,  0.10586993,  0.13905858]), 'zs': jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
# with LogprobCtx(X) as ctx2:
#     gmm(ys)
#     print(ctx2.log_prob)
# exit()

m: Model = model(gmm)(ys)

def find_K(slp: SLP):
    return slp.decision_representative["K"].item()
def formatter(slp: SLP):
    K = find_K(slp) + 1
    return f"#clusters={K}"
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_K)

rng_key = jax.random.PRNGKey(0)

active_slps: List[SLP] = []
for i in tqdm(range(100)):
    rng_key, key = jax.random.split(rng_key)
    X = sample_from_prior(m, key)
    # print(slp.formatted(), slp.branching_decisions.to_human_readable())

    if all(slp.path_indicator(X) == 0 for slp in active_slps):
        slp = slp_from_decision_representative(m, X)
        active_slps.append(slp)
        # print(i, slp.formatted())

        # print(slp.branching_decisions.decisions)
        # slp_to_mcmc_step[slp] = get_inference_regime_mcmc_step_for_slp(slp, deepcopy(regime), config.n_chains, config.collect_intermediate_chain_states)

active_slps = sorted(active_slps, key=m.slp_sort_key)
active_slps = active_slps[:3]

# from dccxjax.infer.estimate_Z import _log_IS_weight_gaussian_mixture
# slp = active_slps[0]
# print(slp.short_repr())
# print(slp.decision_representative)
# X = StackedTrace(broadcast_jaxtree(slp.decision_representative, (4,)), 4)
# _log_IS_weight_gaussian_mixture(slp, jax.random.PRNGKey(0), unstack_trace(X), 1.)


# exit()
# for i, slp in enumerate(active_slps):
#     print(slp.short_repr(), slp.formatted())


n_chains = 10
collect_states = True
collect_infos = True
n_samples_per_chain = 10_000

class CollectType(NamedTuple):
    position: Trace
    log_prob: FloatArray
def return_map(x: MCMCState):
    return CollectType(x.position, x.log_prob) if collect_states else None


from dccxjax.infer.ais import *
def sigmoid(z):
    return 1/(1 + jnp.exp(-z))
def get_Z_ESS(log_weights):
    log_Z = jax.scipy.special.logsumexp(log_weights) - jnp.log(log_weights.shape[0])
    print(f"{log_Z=}")
    Z = jnp.exp(log_Z)
    ESS = jnp.exp(jax.scipy.special.logsumexp(log_weights)*2 - jax.scipy.special.logsumexp(log_weights*2))
    return Z, ESS

from gibbs_proposals import *

def try_estimate_Z_with_AIS():
    for i, slp in enumerate(active_slps):
        print(slp.short_repr(), slp.formatted())

        gibbs_regime = Gibbs(
            InferenceStep(SingleVariable("w"), MH(WProposal(delta, slp.decision_representative["K"].item()))),
            InferenceStep(SingleVariable("mus"), MH(MusProposal(ys, kappa, xi, slp.decision_representative["K"].item()))),
            InferenceStep(SingleVariable("vars"), MH(VarsProposal(ys, alpha, beta, slp.decision_representative["K"].item()))),
            InferenceStep(SingleVariable("zs"), MH(ZsProposal(ys))),
        )

        def w_proposer(w: jax.Array) -> dist.Distribution:
            T = dist.biject_to(dist.Dirichlet(jnp.ones(slp.decision_representative["K"].item())).support)
            w_unconstrained = T.inv(w)
            return dist.TransformedDistribution(dist.Normal(w_unconstrained, 0.5), T)

        regime = Gibbs(
            InferenceStep(SingleVariable("w"), RW(w_proposer)),
            InferenceStep(SingleVariable("mus"), RW(lambda x: dist.Normal(x, 1.0), sparse_numvar=2)),
            InferenceStep(SingleVariable("vars"), RW(lambda x: dist.LeftTruncatedDistribution(dist.Normal(x, 1.0), low=0.), sparse_numvar=2)),
            InferenceStep(SingleVariable("zs"), RW(lambda x: dist.DiscreteUniform(jax.lax.zeros_like_array(x), slp.decision_representative["K"].item()), elementwise=True)),
        )

        mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, gibbs_regime, collect_inference_info=True, return_map=return_map)
        progressbar_mng, mcmc_step = add_progress_bar(n_samples_per_chain, mcmc_step)
        progressbar_mng.start_progress()
        keys = jax.random.split(jax.random.PRNGKey(0), n_samples_per_chain)

        init_info: InferenceInfos = [step.algo.init_info() for step in gibbs_regime] if collect_infos else []
        init = MCMCState(jnp.array(0,int), jnp.array(1.,float), *broadcast_jaxtree((slp.decision_representative, slp.log_prob(slp.decision_representative), init_info), (n_chains,)))

        last_state, all_states = jax.lax.scan(mcmc_step, init, keys)
        last_state.iteration.block_until_ready()
        print("\t", last_state.infos)

        assert all_states is not None
        result_positions = StackedTraces(all_states.position, n_samples_per_chain, n_chains)
        result_positions = result_positions.unstack()

        # harmonic mean of likelihood estimator
        # def log_likelihood(X: Trace):
        #     with LogprobCtx(X) as ctx:
        #         m()
        #         return ctx.log_likelihood
        # log_likeli, _ =  jax.vmap(jax.jit(retrace_branching(log_likelihood, slp.branching_decisions)))(result_positions.data)
        # log_Z = jnp.log(result_positions.n_samples()) - jax.scipy.special.logsumexp(-log_likeli)
        # print(f"{log_Z=}")


        # fig, axs = plt.subplots(1,2, figsize=(12,6))
        # axs[0].hist(result_positions.data["mus"][:,0], bins=100, density=True, label="mu")
        # axs[1].hist(jnp.sqrt(result_positions.data["vars"][:,0]), bins=100, density=True, label="var")
        # plt.show()

        N = 10_000
        tempering_schedule = sigmoid(jnp.linspace(-25,25,1_000))
        tempering_schedule = tempering_schedule.at[0].set(0.)
        tempering_schedule = tempering_schedule.at[-1].set(1.)

        def generate_from_prior_conditioned(rng_key: PRNGKey):
            with GenerateCtx(rng_key, {"K": jnp.array(slp.decision_representative["K"].item(), int)}) as ctx:
                m()
                return ctx.X
            
        X, _ = jax.vmap(jax.jit(retrace_branching(generate_from_prior_conditioned, slp.branching_decisions)))(jax.random.split(jax.random.PRNGKey(0), N))
        lp = jax.vmap(slp.log_prior)(X)

        kernel = get_inference_regime_mcmc_step_for_slp(slp, regime)
        progressbar_mng, kernel = add_progress_bar(tempering_schedule.size, kernel)
        progressbar_mng.start_progress()

        config = AISConfig(None, 0, kernel, tempering_schedule)

        log_weights, position = run_ais(slp, config, jax.random.PRNGKey(0), X, lp, N)
        print(f"{log_weights=}")
        print(get_Z_ESS(log_weights))

        # fig = plt.figure()
        # plt.hist(log_weights, bins=100, density=True)
        
        result_positions = StackedTrace(position, N).unstack()

        # fig, axs = plt.subplots(1,2, figsize=(12,6))
        # axs[0].hist(result_positions.data["mus"][:,0], bins=100, density=True, label="mu")
        # axs[1].hist(jnp.sqrt(result_positions.data["vars"][:,0]), bins=100, density=True, label="var")
        # plt.show()
try_estimate_Z_with_AIS()
exit()


for i, slp in enumerate(active_slps):
    print(slp.short_repr(), slp.formatted())

    Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_prior(slp, 100_000, jax.random.PRNGKey(0))
    print("\t", f" prior Z={Z.item()}, ESS={ESS.item()}, {frac_out_of_support=}")


    regime = Gibbs(
        InferenceStep(SingleVariable("w"), MH(WProposal(delta, slp.decision_representative["K"].item()))),
        InferenceStep(SingleVariable("mus"), MH(MusProposal(ys, kappa, xi, slp.decision_representative["K"].item()))),
        InferenceStep(SingleVariable("vars"), MH(VarsProposal(ys, alpha, beta, slp.decision_representative["K"].item()))),
        InferenceStep(SingleVariable("zs"), MH(ZsProposal(ys))),
    )

    def w_proposer(w: jax.Array) -> dist.Distribution:
        T = dist.biject_to(dist.Dirichlet(jnp.ones(slp.decision_representative["K"].item())).support)
        w_unconstrained = T.inv(w)
        return dist.TransformedDistribution(dist.Normal(w_unconstrained, 0.25), T)
    regime = Gibbs(
        InferenceStep(SingleVariable("w"), RW(w_proposer)),
        InferenceStep(SingleVariable("mus"), RW(lambda x: dist.Normal(x, 1.0), sparse_numvar=2)),
        InferenceStep(SingleVariable("vars"), RW(lambda x: dist.LeftTruncatedDistribution(dist.Normal(x, 1.0), low=0.), sparse_numvar=2)),
        # InferenceStep(SingleVariable("zs"), MH(ZsProposal(ys))),
        InferenceStep(SingleVariable("zs"), RW(lambda x: dist.DiscreteUniform(jax.lax.zeros_like_array(x), slp.decision_representative["K"].item()), elementwise=True)),
    )

    init_info: InferenceInfos = [step.algo.init_info() for step in regime] if collect_infos else []
    init = MCMCState(jnp.array(0,int), jnp.array(1.,float), *broadcast_jaxtree((slp.decision_representative, slp.log_prob(slp.decision_representative), init_info), (n_chains,)))

    mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, regime, collect_inference_info=collect_infos, return_map=return_map)
    progressbar_mng, mcmc_step = add_progress_bar(n_samples_per_chain, mcmc_step)
    progressbar_mng.start_progress()
    keys = jax.random.split(jax.random.PRNGKey(0), n_samples_per_chain)
    last_state, all_states = jax.lax.scan(mcmc_step, init, keys)
    last_state.iteration.block_until_ready()
    print("\t", last_state.infos)

    result_positions = StackedTraces(all_states.position, n_samples_per_chain, n_chains) if all_states is not None else StackedTrace(last_state.position, n_chains)
    result_lps: jax.Array | float = all_states.log_prob if all_states is not None else last_state.log_prob

    amax = jnp.unravel_index(jnp.argmax(result_lps), result_lps.shape)
    map = result_positions.get(*amax)
    print("\t", map)
    y_range = jnp.linspace(ys.min()-2, ys.max()+2, 1000)
    p = jnp.sum(map["w"].reshape(1,-1) * jnp.exp(dist.Normal(map["mus"].reshape(1,-1), jnp.sqrt(map["vars"]).reshape(1,-1)).log_prob(y_range.reshape(-1,1))), axis=1)
    plt.plot(y_range, p, color="gray")
    # plt.scatter(ys, jnp.full_like(ys, 0.))
    K = map["w"].size
    cmap = plt.get_cmap('tab10')
    for k in range(K):
        cluster_ys = ys[map["zs"] == k]
        y_range = jnp.linspace(cluster_ys.min()-2, cluster_ys.max()+2, 1000)
        p = map["w"][k] * jnp.exp(dist.Normal(map["mus"][k], jnp.sqrt(map["vars"])[k]).log_prob(y_range))
        plt.plot(y_range, p, c=cmap(k))
        plt.scatter(cluster_ys, jnp.full_like(cluster_ys, 0.), color=cmap(k))
    plt.show()

    # positions_unstacked = StackedTraces(result_positions, n_samples_per_chain, n_chains).unstack() if collect_states else Traces(result_positions, n_samples_per_chain)
    # positions_unstacked_unconstrained = jax.vmap(slp.transform_to_unconstrained)(positions_unstacked.data)

    # for s in [0.01, 0.1, 1.0,]:
    #     Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_sparse_mixture(slp, s, s, 1.0, 0.0, 1, jax.random.PRNGKey(0), positions_unstacked_unconstrained, True)

    #     print("\t", f" MCMC constrained {s=} Z={Z.item()}, ESS={ESS.item():,.0f}, frac_out_of_support={frac_out_of_support.item()}")

    
# 0.0008
# 0.00038
# 0.00104
# 0.33794
# 0.43936
# 0.17692
# 0.03692
# 0.00624
# 0.0004
# 0.0

t1 = time()

print(f"Total time: {t1-t0:.3f}s")
comp_time = compilation_time_tracker.get_total_compilation_time_secs()
print(f"Total compilation time: {comp_time:.3f}s ({comp_time / (t1 - t0) * 100:.2f}%)")
