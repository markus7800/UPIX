from data import *
from evaluation.gmm.gibbs_proposals import *

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

def chibs(K: int, n_chains:int = 10, n_samples_per_chain: int = 10_000):
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

    # seed, key = jax.random.split(seed)
    # i = jax.random.randint(key, (n_chains,), 0, n_samples_per_chain)
    # j = jnp.arange(n_chains)
    # init_trace = result_positions.get(i,j)
    # init_lp = result_lps[i,j]

    # i = jnp.argmax(result_lps, axis=0)
    # j = jnp.arange(n_chains)
    # init_trace = result_positions.get(i,j)
    # init_lp = result_lps[i,j]

    init_trace = broadcast_jaxtree(map_trace, (n_chains,))
    init_lp = broadcast_jaxtree(map_lp, (n_chains,))

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

def do_chibs(n_chains = 10, n_samples_per_chain = 10_000):
    print(f"do chibs {n_chains=:,} {n_samples_per_chain=:,} {ys.shape=}")

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

    log_Z_path_prior = dist.Poisson(lam-1).log_prob(jnp.array(list(Ks)))
    log_Z = log_Z_path + log_Z_path_prior
    path_weight = jnp.exp(log_Z - jax.scipy.special.logsumexp(log_Z))

    print(f"do chibs {n_chains=:,} {n_samples_per_chain=:,} {ys.shape=}")
    for i, k in enumerate(Ks):
        print(k, path_weight[i])