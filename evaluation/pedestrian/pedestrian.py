import sys
sys.path.insert(0, ".")

from dccxjax import *
import jax
import jax.numpy as jnp
import dccxjax.distributions as dist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List

import logging
setup_logging(logging.WARNING)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)


def pedestrian():
    start = sample("start", dist.Uniform(0.,3.))
    position = start
    distance = 0.
    t = 0
    while (position > 0) & (distance < 10):
        t += 1
        step = sample(f"step_{t}", dist.Uniform(-1.,1.))
        position += step
        distance += jax.lax.abs(step)
    sample("obs", dist.Normal(distance, 0.1), observed=1.1)
    return start


m: Model = model(pedestrian)()

def find_t_max(slp: SLP):
    t_max = 0
    for addr in slp.decision_representative.keys():
        if addr.startswith("step_"):
            t_max = max(t_max, int(addr[5:]))
    return t_max
def formatter(slp: SLP):
    t_max = find_t_max(slp)
    return f"#steps={t_max}"
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_t_max)

rng_key = jax.random.PRNGKey(0)
active_slps: List[SLP] = []
for _ in tqdm(range(1_000)):
    rng_key, key = jax.random.split(rng_key)
    X = sample_from_prior(m, key)
    slp = slp_from_decision_representative(m, X)

    if all(slp.path_indicator(X) == 0 for slp in active_slps):
        active_slps.append(slp)
        # slp_to_mcmc_step[slp] = get_inference_regime_mcmc_step_for_slp(slp, deepcopy(regime), config.n_chains, config.collect_intermediate_chain_states)
active_slps = sorted(active_slps, key=m.slp_sort_key)
active_slps = active_slps[:10]

def distance_position(X: Trace):
    position = X["start"]
    distance = jnp.array(0.)
    for addr, value in X.items():
        if addr.startswith("step"):
            position += value
            distance += jax.lax.abs(value)
    return distance, position

from dccxjax.infer.mcmc import InferenceState, get_inference_regime_mcmc_step_for_slp, add_progress_bar

n_chains = 1_000
for i, slp in enumerate(active_slps):
    print(slp.short_repr(), slp.formatted())
    # print("\t", distance_position(slp.decision_representative), slp.log_prob(slp.decision_representative))
    # position, log_prob, result = coordinate_ascent(slp, 0.1, 10000, n_chains, jax.random.PRNGKey(0))
    # position, log_prob, result = simulated_annealing(slp, 0.1, 10000, n_chains, jax.random.PRNGKey(0))
    p = min(1., 2. / len(slp.decision_representative))
    position, log_prob, result = sparse_coordinate_ascent(slp, 0.1, p, 1_000, n_chains, jax.random.PRNGKey(0))
    # print("\t", distance_position(position), log_prob)
    log_prob.block_until_ready()
    
    n_samples_per_chain = 1_000
    keys = jax.random.split(rng_key, n_samples_per_chain)

    # regime = InferenceStep(AllVariables(), RandomWalk(gaussian_random_walk(0.1), sparse_numvar=2))
    regime = Gibbs(
        InferenceStep(PrefixSelector("step_"), RandomWalk(lambda x: dist.TwoSidedTruncatedDistribution(dist.Normal(x, 0.05), -1.,1.), sparse_numvar=2)),
        InferenceStep(SingleVariable("start"), RandomWalk(lambda x: dist.TwoSidedTruncatedDistribution(dist.Normal(x, 0.05), 0., 3.)))
    )
    mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, regime, n_chains, True)
    mcmc_step = add_progress_bar(n_samples_per_chain, n_chains, mcmc_step)

    init = InferenceState(jax.lax.broadcast(0, (n_chains,)), position)
    last_state, all_positions = jax.lax.scan(mcmc_step, init, keys)
    last_state.iteration.block_until_ready()

    # Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_prior(slp, 10_000_000, jax.random.PRNGKey(0))
    # print("\t", f"prior {Z=}, {ESS=}, {frac_out_of_support=}")
    # all_positions_unstacked = unstack_chains(all_positions)

    # Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_mcmc(slp, 0.1, 10_000_000 // (n_samples_for_unstacked_chains(all_positions_unstacked)), jax.random.PRNGKey(0), all_positions_unstacked)
    # print("\t", f" mcmc {Z=}, {ESS=}, {frac_out_of_support=}")

#     plt.figure()
#     plt.plot(all_positions["start"], alpha=0.5)
# plt.show()

print(f"Total compilation time: {compilation_time_tracker.get_total_compilation_time_secs():.3f}s")

# config = DCC_Config(
#     n_samples_from_prior = 10,
#     n_chains = 4,
#     collect_intermediate_chain_states = True,
#     n_samples_per_chain = 1024,
#     n_samples_for_Z_est = 10**6
# )
# result = dcc(m, lambda _: InferenceStep(AllVariables(), RandomWalk(gaussian_random_walk(0.25), block_update=False)), jax.random.PRNGKey(0), config)

# # plot_histogram(result, "start")
# # plt.show()
# plot_trace(result, "start")
# plt.savefig("tmp1.pdf")
# plot_histogram_by_slp(result, "start")
# plt.savefig("tmp2.pdf")