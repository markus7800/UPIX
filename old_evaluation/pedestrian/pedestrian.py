import sys
sys.path.insert(0, ".")

from dccxjax.all import *
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

active_slps_tree = {}
def maybe_add_new_slp(active_slps_tree, slp: SLP):
    decisions = slp.branching_decisions.decisions
    for i, b in enumerate(decisions):
        assert isinstance(active_slps_tree, dict)
        assert b.shape == () and isinstance(b.item(), (bool, int))
        b = b.item()
        if i == len(decisions) - 1:
            if b in active_slps_tree:
                assert isinstance(active_slps_tree[b], SLP)
                return False
            else:
                active_slps_tree[b] = slp
                active_slps.append(slp)
                return True
        else:
            if b in active_slps_tree:
                active_slps_tree = active_slps_tree[b]
            else:
                active_slps_tree[b] = {}
                active_slps_tree = active_slps_tree[b]
            
    raise Exception


def maybe_add_new_slp_2(active_slps_tree: dict, slp: SLP):
    decisions = slp.branching_decisions.decisions
    t = tuple(x.item() for x in decisions)
    if t not in active_slps_tree:
        active_slps_tree[t] = slp
        active_slps.append(slp)

    
# for _ in tqdm(range(10_000)):
#     rng_key, key = jax.random.split(rng_key)
#     X, decisions = sample_from_prior_with_decisions(m, key)
#     maybe_add_new_slp_2(active_slps_tree, SLP(m, X, decisions))

# from pprint import pprint
# pprint(active_slps_tree)

for _ in tqdm(range(1_000)):
    rng_key, key = jax.random.split(rng_key)
    X = sample_from_prior(m, key)

    if all(slp.path_indicator(X) == 0 for slp in active_slps):
        slp = slp_from_decision_representative(m, X)
        active_slps.append(slp)

print(f"{len(active_slps)=}")


# exit()
# slp = active_slps[4]
# print(slp.get_is_discrete_map())
# X_unconstrained = slp.transform_to_unconstrained(slp.decision_representative)
# print("decision_representative", slp.decision_representative)
# print("X_unconstrained", X_unconstrained)
# lp, X_constrained = slp.unconstrained_log_prob(X_unconstrained)
# print(lp, X_constrained)
# print("X_constrained", slp.transform_to_constrained(X_unconstrained))

# exit()

def distance_position(X: Trace):
    position = X["start"]
    distance = jnp.array(0.)
    for addr, value in X.items():
        if addr.startswith("step"):
            position += value
            distance += jax.lax.abs(value)
    return distance, position

from dccxjax.infer.mcmc import InferenceCarry, InferenceInfos, InferenceState, get_inference_regime_mcmc_step_for_slp, add_progress_bar
from dccxjax.infer.dcc import DCC_Result
from dccxjax.infer.estimate_Z import estimate_Z_for_SLP_from_sparse_mixture

active_slps = sorted(active_slps, key=m.slp_sort_key)
active_slps = active_slps[:10]

collect_states = True
collect_infos = True
combined_result = DCC_Result(collect_states)
n_chains = 10
n_samples_per_chain = 1_000

def return_map(x: InferenceCarry):
    return x.state.position if collect_states else None

for i, slp in enumerate(active_slps):
    print(slp.short_repr(), slp.formatted())
    # print("\t", distance_position(slp.decision_representative), slp.log_prob(slp.decision_representative))
    # position, log_prob, result = coordinate_ascent(slp, 0.1, 10000, n_chains, jax.random.PRNGKey(0))
    # position, log_prob, result = simulated_annealing(slp, 0.1, 10000, n_chains, jax.random.PRNGKey(0))
    p = min(1., 2. / len(slp.decision_representative))
    mle_position, log_prob, result = sparse_coordinate_ascent(slp, 0.1, p, 1_000, 1, jax.random.PRNGKey(0))
    print("\t", f"{mle_position=}")
    print("\t", f"{log_prob=}, {jnp.exp(log_prob)=}")
    print("\t", distance_position(mle_position))
    # position = broadcast_trace(mle_position, (n_chains,))
    position = jax.tree.map(lambda v: jax.lax.broadcast_in_dim(v, (n_chains,)+v.shape[1:], range(len(v.shape))), mle_position)
    log_prob = jax.lax.broadcast_in_dim(log_prob, (n_chains,), [0])
    log_prob.block_until_ready()
    
    rng_key, key = jax.random.split(rng_key)
    keys = jax.random.split(key, n_samples_per_chain)

    # regime = MCMCStep(AllVariables(), RandomWalk(gaussian_random_walk(0.1), sparse_numvar=2))
    regime = MCMCSteps(
        MCMCStep(PrefixSelector("step_"), RandomWalk(lambda x: dist.TwoSidedTruncatedDistribution(dist.Normal(x, 0.2), -1.,1.), sparse_numvar=2)),
        MCMCStep(SingleVariable("start"), RandomWalk(lambda x: dist.TwoSidedTruncatedDistribution(dist.Normal(x, 0.2), 0., 3.)))
    )
    mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, regime, n_chains, collect_infos, return_map)
    progressbar_mng, mcmc_step = add_progress_bar(n_samples_per_chain, n_chains, mcmc_step)

    init_info: InferenceInfos = broadcast_jaxtree([step.algo.init_info() for step in regime] if collect_infos else [], (n_chains,))

    init = InferenceCarry(jax.lax.broadcast(0, (n_chains,)), InferenceState(position, log_prob), init_info)

    progressbar_mng.start_progress(n_samples_per_chain)
    last_state, all_positions = jax.lax.scan(mcmc_step, init, keys)
    last_state.iteration.block_until_ready()
    last_positions = last_state.state.position
    acceptance_rates = jax.tree.map(lambda v: jnp.mean(v) / n_samples_per_chain, last_state.infos)
    print(acceptance_rates)


    # rng_key, key = jax.random.split(rng_key)
    # keys = jax.random.split(key, n_samples_per_chain)
    # progressbar_mng.start_progress(n_samples_per_chain)
    # last_state, all_positions = jax.lax.scan(mcmc_step, init, keys)
    # last_state.iteration.block_until_ready()

    Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_prior(slp, 10_000_000, jax.random.PRNGKey(0))
    print("\t", f" prior Z={Z.item()}, ESS={ESS.item():,.0f}, {1-frac_out_of_support=}")

    Z_final, ESS_final = Z, ESS

    result_positions: Trace =  all_positions if all_positions is not None else last_positions

    positions_unstacked = StackedTraces(result_positions, n_samples_per_chain, n_chains).unstack() if collect_states else Traces(result_positions, n_samples_per_chain)

    # p = 2. / len(slp.decision_representative)
    # p = 1.0
    # for s in [0.1,0.5,1.0,1.5,2.0,5.0,10.0]:
    #     # Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_mcmc(slp, s, 10_000_000 // positions_unstacked.n_samples(), jax.random.PRNGKey(0), Xs_constrained=positions_unstacked.data)
    #     Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_sparse_mixture(slp, s, s, p, 1.0, 10_000_000 // positions_unstacked.n_samples(), jax.random.PRNGKey(0), positions_unstacked.data, False)
    #     if ESS > ESS_final:
    #         Z_final, ESS_final = Z, ESS

    #     print("\t", f" MCMC constrained {s=} Z={Z.item()}, ESS={ESS.item():,.0f}, frac_out_of_support={frac_out_of_support.item()}")

    # positions_unstacked_unconstrained = jax.vmap(slp.transform_to_unconstrained)(positions_unstacked.data)
    # for s in [0.1,0.5,1.0,1.5,2.0,5.0,10.0]:
    #     # Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_mcmc(slp, s, 10_000_000 // positions_unstacked.n_samples(), jax.random.PRNGKey(0), Xs_unconstrained=positions_unstacked_unconstrained)
    #     Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_sparse_mixture(slp, s, s, p, 1.0, 10_000_000 // positions_unstacked.n_samples(), jax.random.PRNGKey(0), positions_unstacked_unconstrained, True)
    #     if ESS > ESS_final:
    #         Z_final, ESS_final = Z, ESS
    #     print("\t", f" MCMC unconstrained {s=} Z={Z.item()}, ESS={ESS.item():,.0f}, frac_out_of_support={frac_out_of_support.item()}")

    combined_result.add_samples(slp, result_positions, Z_final)
#     plt.figure()
#     # plt.plot(all_positions["start"], alpha=0.5)
#     plt.hist(all_positions["start"].reshape(-1), alpha=0.5)
#     plt.title(slp.formatted())

# plt.show()

t1 = time()

print(f"Total time: {t1-t0:.3f}s")
comp_time = compilation_time_tracker.get_total_compilation_time_secs()
print(f"Total compilation time: {comp_time:.3f}s ({comp_time / (t1 - t0) * 100:.2f}%)")

exit()

plot_histogram(combined_result, "start")

qs = jnp.load("evaluation/pedestrian/gt_qs.npy")
ps = jnp.load("evaluation/pedestrian/gt_ps.npy")

fig = plt.gcf()
ax = fig.axes[0]
ax.plot(qs, ps)
plt.show()
# config = DCC_Config(
#     n_samples_from_prior = 10,
#     n_chains = 4,
#     collect_intermediate_chain_states = True,
#     n_samples_per_chain = 1024,
#     n_samples_for_Z_est = 10**6
# )
# result = dcc(m, lambda _: MCMCStep(AllVariables(), RandomWalk(gaussian_random_walk(0.25), block_update=False)), jax.random.PRNGKey(0), config)

# # plot_histogram(result, "start")
# # plt.show()
# plot_trace(result, "start")
# plt.savefig("tmp1.pdf")
# plot_histogram_by_slp(result, "start")
# plt.savefig("tmp2.pdf")