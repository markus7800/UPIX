import sys
sys.path.insert(0, ".")

from dccxjax import *
import jax
import jax.numpy as jnp
import dccxjax.distributions as dist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List
from time import time

import logging
setup_logging(logging.WARNING)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)

t0 = time()

def gmm(ys: jax.Array):
    N = ys.shape[0]
    K = 4# sample("K", dist.Poisson(3)) + 1
    w = sample("w", dist.Dirichlet(jnp.full((K,), 5.0)))
    mus = sample("mus", dist.Normal(jnp.full((K,), 0.0), jnp.full((K,), 1/0.1)))
    vars = sample("vars", dist.InverseGamma(jnp.full((K,), 2.), jnp.full((K,), 10.0)))
    zs = sample("zs", dist.Categorical(jax.lax.broadcast(w, (N,))))
    sample("ys", dist.Normal(mus[zs], vars[zs]), observed=ys)


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

ys = ys[:10]

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
    return 3#slp.decision_representative["K"].item()
def formatter(slp: SLP):
    K = find_K(slp) + 1
    return f"#clusters={K}"
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_K)

rng_key = jax.random.PRNGKey(0)
active_slps: List[SLP] = []
for _ in tqdm(range(100)):
    rng_key, key = jax.random.split(rng_key)
    X = sample_from_prior(m, key)
    slp = slp_from_decision_representative(m, X)
    # print(slp.formatted(), slp.branching_decisions.to_human_readable())

    if all(slp.path_indicator(X) == 0 for slp in active_slps):
        active_slps.append(slp)

        # slp_to_mcmc_step[slp] = get_inference_regime_mcmc_step_for_slp(slp, deepcopy(regime), config.n_chains, config.collect_intermediate_chain_states)


active_slps = sorted(active_slps, key=m.slp_sort_key)
active_slps = active_slps[:10]

for i, slp in enumerate(active_slps):
    print(slp.short_repr(), slp.formatted())

    Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_prior(slp, 100_000, jax.random.PRNGKey(0))
    print("\t", f" prior Z={Z.item()}, ESS={ESS.item()}, {frac_out_of_support=}")

    p = min(1., 2. / len(slp.decision_representative))
    mle_position, log_prob, result = sparse_coordinate_ascent2(slp, 0.1, p, 1_000, 1, jax.random.PRNGKey(0))
    plt.plot(result)
    plt.show()
    print("\t", f"{slp.decision_representative=} log_prob={slp.log_prob(slp.decision_representative)}")
    print("\t", f"MLE {mle_position=} log_prob={log_prob} result={result}")

    for s in [0.01, 0.05, 0.1, 0.5, 1.0,1.5,2.0,]:
        Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_mcmc(slp, s, 10_000_000, jax.random.PRNGKey(0), Xs_constrained=mle_position)
        print("\t", f" MLE constrained {s=} Z={Z.item()}, ESS={ESS.item()}, frac_out_of_support={frac_out_of_support.item()}")

t1 = time()

print(f"Total time: {t1-t0:.3f}s")
comp_time = compilation_time_tracker.get_total_compilation_time_secs()
print(f"Total compilation time: {comp_time:.3f}s ({comp_time / (t1 - t0) * 100:.2f}%)")
