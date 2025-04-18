import sys
sys.path.insert(0, ".")
import os
# import multiprocessing

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
#     multiprocessing.cpu_count()
# )
# os.environ["JAX_PLATFORMS"] = "cpu"

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


import logging
setup_logging(logging.WARN)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)

t0 = time()

lam = 3
delta = 5.0
xi = 0.0
kappa = 0.01
alpha = 2.0
beta = 10.0


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
ys = ys[:5]

def get_lw_log_weight(K):
    @jax.jit
    def lw(rng_key: PRNGKey):
        w_key, mus_key, vars_key = jax.random.split(rng_key,3)

        w = dist.Dirichlet(jnp.full((K+1,), delta)).sample(w_key)
        mus = dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))).sample(mus_key)
        vars = dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).sample(vars_key)

        log_likelihoods = jnp.log(jnp.sum(w.reshape(1,-1) * jnp.exp(dist.Normal(mus.reshape(1,-1), jnp.sqrt(vars).reshape(1,-1)).log_prob(ys.reshape(-1,1))), axis=1))

        return log_likelihoods.sum()
    return lw

def get_lw_log_weight_2(K):
    @jax.jit
    def lw(rng_key: PRNGKey):
        w_key, mus_key, vars_key, zs_key = jax.random.split(rng_key,4)

        w = dist.Dirichlet(jnp.full((K+1,), delta)).sample(w_key)
        mus = dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))).sample(mus_key)
        vars = dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).sample(vars_key)

        zs = dist.CategoricalProbs(w).sample(zs_key, ys.shape)

        log_likelihoods = dist.Normal(mus[zs], jnp.sqrt(vars[zs])).log_prob(ys)

        return log_likelihoods.sum()
    return lw

def IS(log_weight, K: int, N: int, batch_method: int = 1):

    batch_size = 1_000_000
    if N > batch_size and batch_method > 0:
        if batch_method == 1:
            assert N % batch_size == 0
            N_batches = N // batch_size

            def batch(rng_key: PRNGKey):
                keys = jax.random.split(rng_key, batch_size)
                log_Ws = jax.vmap(log_weight)(keys)
                return (jax.scipy.special.logsumexp(log_Ws), jax.scipy.special.logsumexp(log_Ws * 2))

            batch_keys = jax.random.split(jax.random.PRNGKey(0), N_batches)
            log_W, log_W_squared = jax.lax.map(batch, batch_keys)
            log_Z = jax.scipy.special.logsumexp(log_W) - jnp.log(N)
            ESS = jnp.exp(jax.scipy.special.logsumexp(log_W)*2 - jax.scipy.special.logsumexp(log_W_squared))
        else:
            keys = jax.random.split(jax.random.PRNGKey(0), N)
            log_W = jax.lax.map(log_weight, keys, batch_size=batch_size)
            log_Z = jax.scipy.special.logsumexp(log_W) - jnp.log(N)
            ESS = jnp.exp(jax.scipy.special.logsumexp(log_W)*2 - jax.scipy.special.logsumexp(log_W*2))
    else:
        keys = jax.random.split(jax.random.PRNGKey(0), N)
        log_W = jax.vmap(log_weight)(keys)
        log_Z = jax.scipy.special.logsumexp(log_W) - jnp.log(N)
        ESS = jnp.exp(jax.scipy.special.logsumexp(log_W)*2 - jax.scipy.special.logsumexp(log_W*2))
        
    return log_Z, ESS

def do_lw():
    N = 1_000_000_000
    batch_method=1
    print(f"do_lw {N=} {batch_method=}")
    Ks = range(0, 7)
    result = [IS(get_lw_log_weight(K), K, N, batch_method=batch_method) for K in tqdm(Ks)]
    log_Z_path = jnp.array([r[0] for r in result])
    ESS = jnp.array([r[1] for r in result])
    print(f"{log_Z_path=}")
    print(f"{ESS=}")
    # print(f"{jnp.exp(log_Z_path - jax.scipy.special.logsumexp(log_Z_path))}")

    log_Z_path_prior = dist.Poisson(lam).log_prob(jnp.array(list(Ks)))
    log_Z = log_Z_path + log_Z_path_prior
    path_weight = jnp.exp(log_Z - jax.scipy.special.logsumexp(log_Z))

    for i, k in enumerate(Ks):
        print(k, path_weight[i])

# do_lw()

# log_Z=Array(-438.8526, dtype=float32)
# log_Z=Array(-430.81674, dtype=float32)
# log_Z=Array(-431.36102, dtype=float32)

# log_Z_path=Array([-24.367477, -23.827473, -21.73211 , -21.072435, -20.742483,
#        -20.544456, -20.412643], dtype=float32)
# ESS=Array([ 9492778., 11764261.,  6564328., 13200248., 22122286., 32606310.,
#        44187160.], dtype=float32)
# 0 0.0021624845
# 1 0.0111325625
# 2 0.13573469
# 3 0.26253274
# 4 0.27386805
# 5 0.20030616
# 6 0.1142641

from gibbs_proposals import *

class HistogramProposer():

    # minweight from 0 to 1
    def __init__(self, x, min_weigth, bins, kind="uniform"):
        bin_weights, bin_edges = jnp.histogram(x, bins=bins)
        min_weigth = bin_weights.max() * min_weigth
        bin_weights = jax.lax.select(bin_weights < min_weigth, jax.lax.full_like(bin_weights,min_weigth), bin_weights)
        self.bin_weights = bin_weights / bin_weights.sum()
        self.bin_edges = bin_edges
        self.kind = kind

    def sample(self, rng_key: PRNGKey, shape = ()):
        bin_key, unif_key = jax.random.split(rng_key)
        bin = dist.CategoricalProbs(self.bin_weights).sample(bin_key, shape)
        if self.kind == "uniform":
            return dist.Uniform(self.bin_edges[bin], self.bin_edges[bin+1]).sample(unif_key)
        if self.kind == "normal":
            a, b = self.bin_edges[bin], self.bin_edges[bin+1]
            return dist.Normal((a+b)/2, (b-a)/2).sample(unif_key)
        raise Exception


    def log_prob(self, x):
        trailing_shape = tuple([1 for _ in range(len(x.shape))])
        x = x.reshape((1,) + x.shape)
        bin_weights = self.bin_weights.reshape((-1,) + trailing_shape)
        a = self.bin_edges[0:-1].reshape((-1,) + trailing_shape)
        b = self.bin_edges[1:].reshape((-1,) + trailing_shape)

        if self.kind == "uniform":
            return jnp.log(jnp.sum(bin_weights * jnp.exp(dist.Uniform(a, b).log_prob(x)), axis=0))
        if self.kind == "normal":
            return jnp.log(jnp.sum(bin_weights * jnp.exp(dist.Normal((a+b)/2, (b-a)/2).log_prob(x)), axis=0))
        raise Exception


def get_posterior_estimates(K: int, n_chains:int = 100, n_samples_per_chain: int = 10_000):
    gibbs_regime = Gibbs(
        InferenceStep(SingleVariable("w"), MH(WProposal(delta, K))),
        InferenceStep(SingleVariable("mus"), MH(MusProposal(ys, kappa, xi, K))),
        InferenceStep(SingleVariable("vars"), MH(VarsProposal(ys, alpha, beta, K))),
        InferenceStep(SingleVariable("zs"), MH(ZsProposal(ys))),
    )

    class CollectType(NamedTuple):
        position: Trace
        log_prob: FloatArray
    def return_map(x: MCMCState):
        return CollectType(x.position, x.log_prob)

    def gmm():
        N = ys.shape[0]
        w = sample("w", dist.Dirichlet(jnp.full((K+1,), delta)))
        mus = sample("mus", dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))))
        vars = sample("vars", dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)))
        zs = sample("zs", dist.Categorical(jax.lax.broadcast(w, (N,))))
        sample("ys", dist.Normal(mus[zs], jax.lax.sqrt(vars[zs])), observed=ys)


    slp = convert_branchless_model_to_SLP(model(gmm)())

    mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, gibbs_regime, collect_inference_info=False, return_map=return_map)
    progressbar_mng, mcmc_step = add_progress_bar(n_samples_per_chain, mcmc_step)
    progressbar_mng.start_progress()
    keys = jax.random.split(jax.random.PRNGKey(0), n_samples_per_chain)

    init = MCMCState(jnp.array(0,int), jnp.array(1.,float), *broadcast_jaxtree((slp.decision_representative, slp.log_prob(slp.decision_representative), []), (n_chains,)))

    last_state, all_states = jax.lax.scan(mcmc_step, init, keys)
    last_state.iteration.block_until_ready()
    
    result_positions = StackedTraces(all_states.position, n_samples_per_chain, n_chains)
    result_lps: FloatArray = all_states.log_prob

    # amax = jnp.unravel_index(jnp.argmax(result_lps), result_lps.shape)
    # map_trace = result_positions.get(*amax)
    # print(map_trace)


    # result_positions = result_positions.unstack()

    # mus = result_positions.data["mus"].reshape(-1)
    # vars = result_positions.data["vars"].reshape(-1)
    # mus = mus[vars < 25]
    # vars = vars[vars < 25]
    # plt.hist2d(mus,vars,bins=100)
    # plt.show()

    # plt.hist(mus, bins=100, density=True)

    # mu_range = jnp.linspace(mus.min(), mus.max(), 1000)
    # p = jnp.sum(map_trace["w"].reshape(1,-1) * jnp.exp(dist.Normal(map_trace["mus"].reshape(1,-1), jnp.sqrt(map_trace["vars"]).reshape(1,-1)).log_prob(mu_range.reshape(-1,1))), axis=1)
    # plt.plot(mu_range, p)

    # plt.show()


    # x = result_positions.data["vars"].reshape(-1)
    # x = result_positions.data["mus"].reshape(-1)

    # plt.hist(x, bins=100, density=True)


    # hist_proposer = HistogramProposer(x, 0.01, jnp.linspace(x.min(), x.max(), 100), kind="normal")
    # x_sample = hist_proposer.sample(jax.random.PRNGKey(0), (1_000_000,))
    # #print(hist_proposer.log_prob(x_sample).sum())
    # plt.hist(x_sample, bins=100, density=True)
    # xrange = jnp.linspace(x.min(), x.max(), 1000)
    # p = jnp.exp(hist_proposer.log_prob(xrange))
    # plt.plot(xrange, p)
    # plt.show()

    x = result_positions.data["mus"].reshape(-1)
    mu_proposer = HistogramProposer(x, 0.0, jnp.linspace(x.min(), x.max(), 100), kind="normal")

    x = result_positions.data["vars"].reshape(-1)
    var_proposer = HistogramProposer(x, 0.0, jnp.linspace(x.min(), x.max(), 100), kind="uniform")

    return mu_proposer, var_proposer

# get_posterior_estimates(1)

def get_is_log_weight(K, mu_proposer, var_proposer):
    @jax.jit
    def log_weight(rng_key: PRNGKey):
        log_w = 0.
        w_key, mus_key, vars_key = jax.random.split(rng_key,3)

        w = dist.Dirichlet(jnp.full((K+1,), delta)).sample(w_key)

        mus = mu_proposer.sample(mus_key, (K+1,))
        log_w += dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))).log_prob(mus).sum() - mu_proposer.log_prob(mus).sum()
        # jax.debug.print("1 {lp}", lp=log_w)
        vars = var_proposer.sample(vars_key, (K+1,))
        # vars = jnp.maximum(vars, 1e-5)

        log_w += dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).log_prob(vars).sum() - var_proposer.log_prob(vars).sum()
        # jax.debug.print("2 {lp}", lp=log_w)
        log_likelihoods = jnp.log(jnp.sum(w.reshape(1,-1) * jnp.exp(dist.Normal(mus.reshape(1,-1), jnp.sqrt(vars).reshape(1,-1)).log_prob(ys.reshape(-1,1))), axis=1))

        return log_w + log_likelihoods.sum()
    return log_weight

def get_is_log_weight_2(K, mu_proposer, var_proposer):
    @jax.jit
    def log_weight(rng_key: PRNGKey):
        log_w = 0.
        w_key, mus_key, vars_key, zs_key = jax.random.split(rng_key,4)

        w = dist.Dirichlet(jnp.full((K+1,), delta)).sample(w_key)

        mus = mu_proposer.sample(mus_key, (K+1,))
        log_w += dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))).log_prob(mus).sum() - mu_proposer.log_prob(mus).sum()
        # jax.debug.print("1 {lp}", lp=log_w)
        vars = var_proposer.sample(vars_key, (K+1,))
        # vars = jnp.maximum(vars, 1e-5)

        log_w += dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).log_prob(vars).sum() - var_proposer.log_prob(vars).sum()

        zs_proposer = ZsProposal(ys)
        zs, zs_lp = zs_proposer.propose(zs_key, {"w": w, "mus": mus, "vars": vars})
        zs = zs["zs"]

        log_w += dist.Categorical(jax.lax.broadcast(w, (ys.shape[0],))).log_prob(zs).sum() - zs_lp

        # jax.debug.print("2 {lp}", lp=log_w)
        log_w += dist.Normal(mus[zs], jnp.sqrt(vars[zs])).log_prob(ys).sum()

        # jax.debug.print("{x} {y} {z} {zs}",
        #     x=dist.Categorical(jax.lax.broadcast(w, (ys.shape[0],))).log_prob(zs).sum(),
        #     y=zs_lp, z=dist.Normal(mus[zs], jnp.sqrt(vars[zs])).log_prob(ys).sum(),
        #     zs=zs,
        #     #d=zs_proposer.get_categorical({"w": w, "mus": mus, "vars": vars}).probs
        # )

        return log_w 
    return log_weight

def do_is():
    N = 1_000_000_000
    print(f"do_is {N=}")
    Ks = range(0, 7)
    result = []
    for K in Ks:
        print(f"{K=}")
        mu_proposer, var_proposer = get_posterior_estimates(K)
        r = IS(get_is_log_weight(K, mu_proposer, var_proposer), K, N, batch_method=1)
        result.append(r)
        print(r)

    log_Z_path = jnp.array([r[0] for r in result])
    ESS = jnp.array([r[1] for r in result])
    print(f"{log_Z_path=}")
    print(f"{ESS=}")
    # print(f"{jnp.exp(log_Z_path - jax.scipy.special.logsumexp(log_Z_path))}")

    log_Z_path_prior = dist.Poisson(lam).log_prob(jnp.array(list(Ks)))
    log_Z = log_Z_path + log_Z_path_prior
    path_weight = jnp.exp(log_Z - jax.scipy.special.logsumexp(log_Z))

    for i, k in enumerate(Ks):
        print(k, path_weight[i])

# do_is()

# N = 1_000_000_000
# log_Z_path=Array([-438.7951 , -412.87714, -378.07547, -372.03406, -371.0071 ,
#        -371.09848, -371.74484], dtype=float32)
# ESS=Array([9.8786797e+08, 1.5797326e+06, 1.3217342e+05, 1.7380820e+04,
#        4.0514241e+03, 4.1891110e+02, 2.2887285e+02], dtype=float32)
# 0 4.9588643e-31
# 1 2.6824395e-19
# 2 0.0005233622
# 3 0.22006674
# 4 0.46090347
# 5 0.25239247
# 6 0.06612039


def chibs(K: int, n_chains:int = 100, n_samples_per_chain: int = 10_000):
    inference_steps = [
        InferenceStep(SingleVariable("w"), MH(WProposal(delta, K))),
        InferenceStep(SingleVariable("mus"), MH(MusProposal(ys, kappa, xi, K))),
        InferenceStep(SingleVariable("vars"), MH(VarsProposal(ys, alpha, beta, K))),
        InferenceStep(SingleVariable("zs"), MH(ZsProposal(ys))),
    ]
    gibbs_regime = Gibbs(*inference_steps)

    class CollectType(NamedTuple):
        position: Trace
        log_prob: FloatArray
    def return_map(x: MCMCState):
        return CollectType(x.position, x.log_prob)

    def gmm():
        N = ys.shape[0]
        w = sample("w", dist.Dirichlet(jnp.full((K+1,), delta)))
        mus = sample("mus", dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))))
        vars = sample("vars", dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)))
        zs = sample("zs", dist.Categorical(jax.lax.broadcast(w, (N,))))
        sample("ys", dist.Normal(mus[zs], jax.lax.sqrt(vars[zs])), observed=ys)


    slp = convert_branchless_model_to_SLP(model(gmm)())

    mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, gibbs_regime, collect_inference_info=False, return_map=return_map)
    progressbar_mng, mcmc_step = add_progress_bar(n_samples_per_chain, mcmc_step)
    progressbar_mng.start_progress()
    keys = jax.random.split(jax.random.PRNGKey(0), n_samples_per_chain)

    init = MCMCState(jnp.array(0,int), jnp.array(1.,float), *broadcast_jaxtree((slp.decision_representative, slp.log_prob(slp.decision_representative), []), (n_chains,)))

    last_state, all_states = jax.lax.scan(mcmc_step, init, keys)
    last_state.iteration.block_until_ready()
    
    result_positions = StackedTraces(all_states.position, n_samples_per_chain, n_chains)
    result_lps: FloatArray = all_states.log_prob

    amax = jnp.unravel_index(jnp.argmax(result_lps), result_lps.shape)
    map_trace = result_positions.get(*amax)
    map_lp = result_lps[amax] # joint
    # print(map_trace, map_lp)


    # n_chains = 2
    # n_samples_per_chain = 10
    seed = jax.random.PRNGKey(0)

    seed, key = jax.random.split(seed)
    i = jax.random.randint(key, (n_chains,), 0, n_samples_per_chain)
    j = jnp.arange(n_chains)
    init_trace = result_positions.get(i,j)
    init_lp = result_lps[i,j]

    # i = jnp.argmax(result_lps, axis=0)
    # j = jnp.arange(n_chains)
    # init_trace = result_positions.get(i,j)
    # init_lp = result_lps[i,j]

    # init_trace = broadcast_jaxtree(map_trace, (n_chains,))
    # init_lp = broadcast_jaxtree(map_lp, (n_chains,))

    names = ["w", "mus", "vars", "zs"]
    log_posterior_est = jnp.zeros(n_chains)
    print("init_lp:", init_lp)

    for i, name in enumerate(names):
        condition_names = [names[j] for j in range(i)]
        sample_names = [names[j] for j in range(i+1, len(names))]
        s = ", ".join([n + "*" for n in condition_names] + sample_names)
        print(f"{name}* | {s}")
        # gibbs targets (sample_names + [name]) | condition_names 
        # regime consists only of steps for sample_names and values of condition_names are initialised to map and keep their values
        # name is also perturbed by regime but marginalised out in density estimation

        regime = Gibbs(*inference_steps[i:])
        mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, regime, collect_inference_info=False, return_map=return_map)
        progressbar_mng, mcmc_step = add_progress_bar(n_samples_per_chain, mcmc_step)
        progressbar_mng.start_progress()

        seed, mcmc_key = jax.random.split(seed)
        keys = jax.random.split(mcmc_key, n_samples_per_chain)
        init = MCMCState(jnp.array(0,int), jnp.array(1.,float), init_trace, init_lp, broadcast_jaxtree([], (n_chains,)))

        last_state, all_states = jax.lax.scan(mcmc_step, init, keys)
        last_state.iteration.block_until_ready()
        
        result_positions = StackedTraces(all_states.position, n_samples_per_chain, n_chains)
        
        proposal = inference_steps[i].algo.proposal
        def gibbs_lp(X: Trace, trace: Trace):
            return jax.vmap(proposal.assess, (0, None))(X, trace)

        glp = jax.vmap(gibbs_lp, in_axes=(1,0))(result_positions.data, init_trace) # (n_chains, n_samples_per_chain)
        log_posterior_est += jax.scipy.special.logsumexp(glp,axis=1) - jnp.log(n_samples_per_chain)


    log_Z_est = init_lp - log_posterior_est
    print("log_Z =", log_Z_est, "+/-", jnp.std(log_Z_est))
    return jax.scipy.special.logsumexp(log_Z_est) - jnp.log(n_chains)

def do_chibs():
    n_chains = 10
    n_samples_per_chain = 100_000
    print(f"do chibs {n_chains=} {n_samples_per_chain=}")

    Ks = range(0, 7)
    result = []
    for K in Ks:
        print(f"{K=}")
        r = chibs(K, n_chains, n_samples_per_chain)
        result.append(r)
        print(r)

    log_Z_path = jnp.array(result)
    print(f"{log_Z_path=}")
    # print(f"{jnp.exp(log_Z_path - jax.scipy.special.logsumexp(log_Z_path))}")

    log_Z_path_prior = dist.Poisson(lam).log_prob(jnp.array(list(Ks)))
    log_Z = log_Z_path + log_Z_path_prior
    path_weight = jnp.exp(log_Z - jax.scipy.special.logsumexp(log_Z))

    for i, k in enumerate(Ks):
        print(k, path_weight[i])

do_chibs()