from dccxjax import *
import jax
import jax.numpy as jnp
import dccxjax.distributions as dist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List
from time import time

from dccxjax.infer.mcmc import InferenceState, get_inference_regime_mcmc_step_for_slp, add_progress_bar
from dccxjax.infer.dcc import DCC_Result

import logging
setup_logging(logging.WARNING)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)

t0 = time()

def normal():
    x = sample("x", dist.Normal(0.,1.))
    sample("y", dist.Normal(x, 0.01), observed=5.)

m: Model = model(normal)()


t1 = time()

slp = convert_branchless_model_to_SLP(m)
regime = InferenceStep(SingleVariable("x"), RW(gaussian_random_walk(0.1)))

n_chains = 100
collect_states = True
n_samples_per_chain = 10_000
mcmc_step = get_inference_regime_mcmc_step_for_slp(slp, regime, n_chains, collect_states)


init = broadcast_jaxtree(InferenceState(0, slp.decision_representative, slp.log_prob(slp.decision_representative)), (n_chains,))

keys = jax.random.split(jax.random.PRNGKey(0), n_samples_per_chain)
last_state, all_positions = jax.lax.scan(mcmc_step, init, keys)
last_state.iteration.block_until_ready()
last_positions = last_state.position


Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_prior(slp, 10_000_000, jax.random.PRNGKey(0))
print("\t", f" prior Z={Z.item()}, ESS={ESS.item()}, {frac_out_of_support=}")

result_positions: Trace =  all_positions if all_positions is not None else last_positions
positions_unstacked = unstack_chains(result_positions) if collect_states else result_positions

s = 1.0
Z, ESS, frac_out_of_support = estimate_Z_for_SLP_from_mcmc(slp, s, 10_000_000 // (n_samples_for_unstacked_chains(positions_unstacked)), jax.random.PRNGKey(0), Xs_constrained=positions_unstacked)
print("\t", f" MCMC constrained {s=} Z={Z.item()}, ESS={ESS.item()}, frac_out_of_support={frac_out_of_support.item()}")
Z.block_until_ready()


x = result_positions["x"]
print(x.shape)
T, N = x.shape
M = 10_000_000 // (T*N)
print(f"{N=} {T=} {M=}")

@jax.jit
def weight_t(x_t, rng_key: PRNGKey):
    # x_t shape = (N, )
    print(x_t)
    print(rng_key)
    
    scale = 1.0
    z = jax.random.normal(rng_key, (N,M))
    x_tilde = x_t.reshape((N,1)) + scale * z # shape = (N,M)
    x_tilde = x_tilde.reshape(N*M)
    log_gamma = jax.vmap(slp.log_prob)({"x": x_tilde})
    log_q = jax.scipy.stats.norm.logpdf(x_tilde.reshape(1,N*M), x_t.reshape(N,1), scale) # shape = (N,N*M)
    log_q = jax.scipy.special.logsumexp(log_q, axis=0) - jax.lax.log(float(N)) # shape (N*M,)
    print(log_q)
    return log_gamma - log_q



    
keys = jax.random.split(jax.random.PRNGKey(0), T)
log_w = jax.vmap(weight_t)(x, keys)
log_w = log_w.reshape(-1)

log_weights_sum = jax.scipy.special.logsumexp(log_w)
log_weights_squared_sum = jax.scipy.special.logsumexp(log_w * 2)
log_ess = log_weights_sum * 2 - log_weights_squared_sum
Z = jnp.exp(log_weights_sum) / (N*T*M)
ess = jnp.exp(log_ess)
print(f"{Z=} {ess=}")



t1 = time()

print(f"Total time: {t1-t0:.3f}s")
comp_time = compilation_time_tracker.get_total_compilation_time_secs()
print(f"Total compilation time: {comp_time:.3f}s ({comp_time / (t1 - t0) * 100:.2f}%)")
