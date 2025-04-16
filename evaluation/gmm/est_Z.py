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


def lw(K: int, N: int):
    @jax.jit
    def log_weight(rng_key: PRNGKey):
        w_key, mus_key, vars_key = jax.random.split(rng_key,3)

        w = dist.Dirichlet(jnp.full((K+1,), delta)).sample(w_key)
        mus = dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))).sample(mus_key)
        vars = dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).sample(vars_key)

        log_likelihoods = jnp.log(jnp.sum(w.reshape(1,-1) * jnp.exp(dist.Normal(mus.reshape(1,-1), jnp.sqrt(vars).reshape(1,-1)).log_prob(ys.reshape(-1,1))), axis=1))

        return log_likelihoods.sum()

    keys =jax.random.split(jax.random.PRNGKey(0), N)
    log_W = jax.lax.map(log_weight, keys, batch_size=min(N,10_000_000))
    log_Z = jax.scipy.special.logsumexp(log_W) - jnp.log(N)

    ESS = jnp.exp(jax.scipy.special.logsumexp(log_W)*2 - jax.scipy.special.logsumexp(log_W*2))
    # print(f"{K}:", log_Z)
    return log_Z, ESS


N = 10_000
Ks = range(0, 7)
result = [lw(K, N) for K in tqdm(Ks)]
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

# log_Z=Array(-438.8526, dtype=float32)
# log_Z=Array(-430.81674, dtype=float32)
# log_Z=Array(-431.36102, dtype=float32)


from gibbs_proposals import *

def get_posterior_estimates(K: int):
    gibbs_regime = Gibbs(
        InferenceStep(SingleVariable("w"), MH(WProposal(delta, K))),
        InferenceStep(SingleVariable("mus"), MH(MusProposal(ys, kappa, xi, K))),
        InferenceStep(SingleVariable("vars"), MH(VarsProposal(ys, alpha, beta, K))),
        InferenceStep(SingleVariable("zs"), MH(ZsProposal(ys))),
    )

    n_chains = 1
    n_samples_per_chain = 100_000

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
    map = result_positions.get(*amax)
    print(map)

    result_positions = result_positions.unstack()
    mus = result_positions.data["mus"]
    mus = jnp.sort(mus, axis=1)

    fig, axs = plt.subplots(K, 2)
    for k in range(K):
        axs[k,0].hist(mus[:,k], bins=100, density=True)
        # axs[k,1].hist(jnp.sqrt(result_positions.data["vars"][:,k]), bins=100, density=True)
    plt.show()


get_posterior_estimates(4)