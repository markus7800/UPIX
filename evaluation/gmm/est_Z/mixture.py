import jax.scipy.special
from dccxjax import *
import numpyro.distributions as dist
from data import *


class MixtureProposer:
    def __init__(self, centers: Trace, N: int, sigma: float, K: int) -> None:
        self.centers = centers
        self.N = N
        self.sigma = sigma
        self.K = K

    def sample(self, rng_key: PRNGKey):
        select_key, w_key, mu_key, vars_key = jax.random.split(rng_key, 4)
        K = self.K
        w_bij = dist.biject_to(dist.Dirichlet(jnp.full((K+1,), delta)).support)
        vars_bij = dist.biject_to(dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).support)
        
        n = dist.DiscreteUniform(0,self.N).sample(select_key)
        center = jax.tree_map(lambda v: v[n,...], self.centers)

        w = dist.TransformedDistribution(dist.Normal(w_bij.inv(center["w"]),self.sigma), w_bij).sample(w_key)
        mus = dist.Normal(center["mus"],self.sigma).sample(mu_key)
        vars = dist.TransformedDistribution(dist.Normal(vars_bij.inv(center["vars"]),self.sigma), vars_bij).sample(vars_key)
        
        return w, mus, vars
    
    def log_prob(self, w, mus, vars):
        K = self.K
        w_bij = dist.biject_to(dist.Dirichlet(jnp.full((K+1,), delta)).support)
        vars_bij = dist.biject_to(dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).support)

        def _lp(lp, center: Trace):
            lp_w = dist.TransformedDistribution(dist.Normal(w_bij.inv(center["w"]),self.sigma), w_bij).log_prob(w)
            lp_mus = dist.Normal(center["mus"],self.sigma).log_prob(mus).sum()
            lp_vars = dist.TransformedDistribution(dist.Normal(vars_bij.inv(center["vars"]),self.sigma), vars_bij).log_prob(vars).sum()
            # print(f"{mus=} {center['mus']=} {lp_mus=}")
            return jnp.logaddexp(lp, lp_w + lp_mus + lp_vars), lp_w + lp_mus + lp_vars
        
        res = jax.lax.scan(_lp, -jnp.inf, self.centers)
        # jax.debug.print("{r1} vs {r2}", r1=res[0], r2=jax.scipy.special.logsumexp(res[1]))
        return res[0] - jnp.log(self.N)
    


def get_is_log_weight(K, proposer: MixtureProposer):
    @jax.jit
    def log_weight(rng_key: PRNGKey):
        log_w = 0.

        w, mus, vars = proposer.sample(rng_key)

        log_w = dist.Dirichlet(jnp.full((K+1,), delta)).log_prob(w)

        log_w += dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))).log_prob(mus).sum()

        log_w += dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).log_prob(vars).sum()

        log_w -= proposer.log_prob(w, mus, vars)
        # jax.debug.print("{l}", l=proposer.log_prob(w, mus, vars))

        # log_likelihoods = jnp.log(jnp.sum(w.reshape(1,-1) * jnp.exp(dist.Normal(mus.reshape(1,-1), jnp.sqrt(vars).reshape(1,-1)).log_prob(ys.reshape(-1,1))), axis=1))
        log_likelihoods = jax.scipy.special.logsumexp((jnp.log(w).reshape(1,-1) + dist.Normal(mus.reshape(1,-1), jnp.sqrt(vars).reshape(1,-1)).log_prob(ys.reshape(-1,1))), axis=1)


        return log_w + log_likelihoods.sum()
    
    return log_weight

from lw import IS

from evaluation.gmm.gibbs_proposals import *
from typing import NamedTuple
from dccxjax.infer.mcmc import MCMCState, InferenceInfos, get_inference_regime_mcmc_step_for_slp, add_progress_bar
import matplotlib.pyplot as plt
from matplotlib import transforms as plt_transforms

def get_posterior(K: int, n_chains:int = 10, n_samples_per_chain: int = 10_000):
    inference_steps = [
        MCMCStep(SingleVariable("w"), MH(WProposal(delta, K))),
        MCMCStep(SingleVariable("mus"), MH(MusProposal(ys, kappa, xi, K))),
        MCMCStep(SingleVariable("vars"), MH(VarsProposal(ys, alpha, beta, K))),
        MCMCStep(SingleVariable("zs"), MH(ZsProposal(ys))),
    ]
    gibbs_regime = MCMCSteps(*inference_steps)

    class CollectType(NamedTuple):
        position: Trace
        log_prob: FloatArray
    def return_map(x: MCMCState):
        return CollectType({"w": x.position["w"], "mus": x.position["mus"], "vars": x.position["vars"]}, x.log_prob)

    def gmm():
        N = ys.shape[0]
        w = sample("w", dist.Dirichlet(jnp.full((K+1,), delta)))
        mus = sample("mus", dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))))
        vars = sample("vars", dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)))
        zs = sample("zs", dist.Categorical(jax.lax.broadcast(w, (N,))))
        sample("ys", dist.Normal(mus[zs], jax.lax.sqrt(vars[zs])), observed=ys)

    slp = SLP_from_branchless_model(model(gmm)())

    mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, gibbs_regime, collect_inference_info=False, return_map=return_map)
    progressbar_mng, mcmc_step = add_progress_bar(n_samples_per_chain, mcmc_step)
    progressbar_mng.start_progress()
    keys = jax.random.split(jax.random.PRNGKey(0), n_samples_per_chain)

    init = MCMCState(jnp.array(0,int), jnp.array(1.,float), *broadcast_jaxtree((slp.decision_representative, slp.log_prob(slp.decision_representative), []), (n_chains,)))

    last_state, all_states = jax.lax.scan(mcmc_step, init, keys)
    last_state.iteration.block_until_ready()
    
    result_positions = StackedTraces(all_states.position, n_samples_per_chain, n_chains).unstack()

    # ixs = jax.random.randint(jax.random.PRNGKey(0), (10_000,), 0, result_positions.n_samples())
    # centers = jax.tree_map(lambda v: v[ixs,...], result_positions.data)

    # plt.scatter(centers["mus"], centers["vars"], alpha=0.1, s=1.)
    # plt.xlabel("mu")
    # plt.ylabel("var")

    # w = centers["w"]
    # w_bij = dist.biject_to(dist.Dirichlet(jnp.full((K+1,), delta)).support)
    # w = jax.vmap(w_bij.inv)(w)

    # if w.shape[1] > 0:
    #     n_plots = w.shape[1] + 1
    #     fig, axs = plt.subplots(n_plots, n_plots, sharex="col", sharey="row")
    #     for i in range(n_plots):
    #         for j in range(n_plots):
    #             if i == n_plots-1:
    #                 if j == n_plots-1:
    #                     continue
    #                 # axs[i,j].hist(w[:,j], density=True, bins=100)
    #             elif j == n_plots-1:
    #                 # axs[i,j].hist(w[:,i], orientation="horizontal", density=True, bins=100)
    #                 continue
    #             else:
    #                 axs[i,j].scatter(w[:,i], w[:,j], alpha=0.1, s=1.)
    #                 # axs[i,j].set_xticklabels([])
    #                 # axs[i,j].set_yticklabels([])
    #     plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()

    
    return result_positions

def do_mixture_is(N = 1_000_000, n_chains:int = 10, n_samples_per_chain: int = 10_000, n_components: int = 1000, sigma: float = 1.0):
    print(f"do_is {N=:,} {ys.shape=}")
    Ks = range(0, 7)
    result = []
    for K in Ks:
        print(f"{K=}")
        traces = get_posterior(K, n_chains, n_samples_per_chain)
        ixs = jax.random.randint(jax.random.PRNGKey(0), (n_components,), 0, traces.n_samples())
        centers = jax.tree_map(lambda v: v[ixs,...], traces.data)
        proposer = MixtureProposer(centers, n_components, sigma, K)

        r = IS(get_is_log_weight(K, proposer), K, N, batch_method=1)
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