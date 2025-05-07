
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

from lw import IS

def lis(K: int, n_chains:int = 10, n_samples_per_chain: int = 10_000):
    inference_steps = [
        MCMCStep(SingleVariable("w"), MH(WProposal(delta, K))),
        MCMCStep(SingleVariable("mus"), MH(MusProposal(ys, kappa, xi, K))),
        MCMCStep(SingleVariable("vars"), MH(VarsProposal(ys, alpha, beta, K))),
        MCMCStep(SingleVariable("zs"), MH(ZsProposal(ys))),
    ]
    regime = MCMCSteps(*inference_steps)

    # def w_proposer(w: jax.Array) -> dist.Distribution:
    #     T = dist.biject_to(dist.Dirichlet(jnp.ones(K)).support)
    #     w_unconstrained = T.inv(w)
    #     return dist.TransformedDistribution(dist.Normal(w_unconstrained, 0.25), T)
    # regime = MCMCSteps(
    #     MCMCStep(SingleVariable("w"), RW(w_proposer)),
    #     MCMCStep(SingleVariable("mus"), RW(lambda x: dist.Normal(x, 1.0), sparse_numvar=2)),
    #     MCMCStep(SingleVariable("vars"), RW(lambda x: dist.LeftTruncatedDistribution(dist.Normal(x, 1.0), low=0.), sparse_numvar=2)),
    #     # MCMCStep(SingleVariable("zs"), RW(lambda x: dist.DiscreteUniform(jax.lax.zeros_like_array(x), K), elementwise=True)),
    # )



    class CollectType(NamedTuple):
        position: Trace
        log_prob: FloatArray
    def return_map(x: MCMCState):
        return CollectType(x.position, x.log_prob)

    def gmm():
        w = sample("w", dist.Dirichlet(jnp.full((K+1,), delta)))
        mus = sample("mus", dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))))
        vars = sample("vars", dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)))
        zs = sample("zs", dist.Categorical(jax.lax.broadcast(w, (ys.shape[0],))))
        sample("ys", dist.Normal(mus[zs], jax.lax.sqrt(vars[zs])), observed=ys)
    
    # def gmm():
    #     w = sample("w", dist.Dirichlet(jnp.full((K+1,), delta)))
    #     mus = sample("mus", dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))))
    #     vars = sample("vars", dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)))

    #     log_likelihoods = jax.scipy.special.logsumexp((jnp.log(w).reshape(1,-1) + dist.Normal(mus.reshape(1,-1), jnp.sqrt(vars).reshape(1,-1)).log_prob(ys.reshape(-1,1))), axis=1)

    #     logfactor(log_likelihoods.sum())
    
    slp = convert_branchless_model_to_SLP(model(gmm)())

    collect_info = True

    mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, regime, collect_inference_info=collect_info, return_map=return_map)
    progressbar_mng, mcmc_step = add_progress_bar(n_samples_per_chain, mcmc_step)
    progressbar_mng.start_progress()
    keys = jax.random.split(jax.random.PRNGKey(0), n_samples_per_chain)

    init = MCMCState(jnp.array(0,int), jnp.array(1.,float), *broadcast_jaxtree((slp.decision_representative, slp.log_prob(slp.decision_representative), init_inference_infos(regime) if collect_info else []), (n_chains,)))

    last_state, all_states = jax.lax.scan(mcmc_step, init, keys)
    last_state.iteration.block_until_ready()
    # print(last_state.infos)

    result_positions = StackedTraces(all_states.position, n_samples_per_chain, n_chains)
    result_lps: FloatArray = all_states.log_prob

    amax = jnp.unravel_index(jnp.argmax(result_lps), result_lps.shape)
    map_trace = result_positions.get(*amax)
    map_lp = result_lps[amax] # joint
    print(map_trace, map_lp)

    traces = result_positions.unstack().data

    w_bij = dist.biject_to(dist.Dirichlet(jnp.full((K+1,), delta)).support)
    vars_bij = dist.biject_to(dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).support)
    
    for s in [0.1,0.5,1.,2.,5.]:

        c = (
            (jnp.abs(w_bij.inv(traces["w"]) - w_bij.inv(jax.lax.broadcast(map_trace["w"], (1,)))) < s).all(axis=1) &
            (jnp.abs(traces["mus"] - jax.lax.broadcast(map_trace["mus"], (1,)) < s)).all(axis=1) &
            (jnp.abs(vars_bij.inv(traces["vars"]) - vars_bij.inv(jax.lax.broadcast(map_trace["vars"], (1,)))) < s).all(axis=1)
        )
        Z_B = jnp.mean(c)
        print(f"{s=} {c.shape} {c.sum()} Z_B={Z_B}")

        w_proposer = dist.TransformedDistribution(dist.Uniform(w_bij.inv(map_trace["w"])-s, w_bij.inv(map_trace["w"])+s, validate_args=False), w_bij)
        mus_proposer = dist.Uniform(map_trace["mus"]-s, map_trace["mus"]+s, validate_args=False)
        vars_proposer = dist.TransformedDistribution(dist.Uniform(vars_bij.inv(map_trace["vars"])-s, vars_bij.inv(map_trace["vars"])+s, validate_args=False), vars_bij)

        @jax.jit
        def log_weight(rng_key: PRNGKey):
            log_w = 0.
            w_key, mus_key, vars_key = jax.random.split(rng_key,3)

            w = w_proposer.sample(w_key)
            log_w += dist.Dirichlet(jnp.full((K+1,), delta)).log_prob(w).sum() - w_proposer.log_prob(w).sum()

            mus = mus_proposer.sample(mus_key)
            log_w += dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))).log_prob(mus).sum() - mus_proposer.log_prob(mus).sum()

            vars = vars_proposer.sample(vars_key)
            log_w += dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).log_prob(vars).sum() - vars_proposer.log_prob(vars).sum()

            # log_likelihoods = jnp.log(jnp.sum(w.reshape(1,-1) * jnp.exp(dist.Normal(mus.reshape(1,-1), jnp.sqrt(vars).reshape(1,-1)).log_prob(ys.reshape(-1,1))), axis=1))
            log_likelihoods = jax.scipy.special.logsumexp((jnp.log(w).reshape(1,-1) + dist.Normal(mus.reshape(1,-1), jnp.sqrt(vars).reshape(1,-1)).log_prob(ys.reshape(-1,1))), axis=1)


            return log_w + log_likelihoods.sum()
        
        logZ, ESS = IS(log_weight, K, 10_000_000, batch_method=1)
        

        print(f"logZ={logZ}, ESS={ESS}")
        print(logZ - jnp.log(Z_B))