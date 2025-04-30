from evaluation.gmm.gibbs_proposals import *
from data import *

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

class HistogramProposer():

    # minweight from 0 to 1
    def __init__(self, x, min_weigth, bins, kind="uniform"):
        bin_weights, bin_edges = jnp.histogram(x, bins=bins) # failed to parse int literal in cuda backend here
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

from lw import IS

def do_is(N = 100_000_000):
    print(f"do_is {N=:,} {ys.shape=}")
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

    log_Z_path_prior = dist.Poisson(lam-1).log_prob(jnp.array(list(Ks)))
    log_Z = log_Z_path + log_Z_path_prior
    path_weight = jnp.exp(log_Z - jax.scipy.special.logsumexp(log_Z))

    print(f"do_is {N=:,} {ys.shape=}")
    for i, k in enumerate(Ks):
        print(k, path_weight[i])