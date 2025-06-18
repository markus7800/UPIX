#%%
import sys
import os
sys.path.insert(0, ".")

from dccxjax import *
import jax
import jax.numpy as jnp
import dccxjax.distributions as dist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List
from time import time
from dccxjax.infer.mcmc.mcmc import MCMCState, CarryStats

import logging
setup_logging(logging.WARNING)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)



@model
def geometric(p: FloatArrayLike, obs: FloatArrayLike):
    b = True
    i = 0
    while b:
        b = sample(f"b_{i}", dist.Bernoulli(p))
        i += 1
    sample("x", dist.Normal(i, 1.0), observed=obs)

m = geometric(0.5, 5.)


def find_i_max(X: Trace):
    i = 0
    for addr in X.keys():
        if addr[0] == "b":
            i = max(i, int(addr[2:]))
    return i

def formatter(slp: SLP):
    i_max = find_i_max(slp.decision_representative)
    return f"i={i_max}"

m.set_slp_formatter(formatter)
m.set_slp_sort_key(lambda slp: find_i_max(slp.decision_representative))

class DCCConfig(MCMCDCC[T]):
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        return MCMCSteps()
    def run_inference(self, slp, rng_key) -> InferenceResult:
        log_prob = slp.log_prob(slp.decision_representative)
        return MCMCInferenceResult(
            None,
            MCMCState(jnp.array(0,int), jnp.array(1.,float), dict(), broadcast_jaxtree(slp.decision_representative, (1,)), log_prob, CarryStats(), None),
            1,
            1,
            False
        )
    def estimate_log_weight(self, slp, rng_key):
        # super().estimate_log_weight(slp, rng_key)
        log_prob_trace = self.model.log_prob_trace(slp.decision_representative)
        log_weight = sum((lp for addr, (lp,observed) in log_prob_trace.items() if addr[0] == "b" or observed), start=jnp.array(0.,float))
        if self.verbose >= 2:
            tqdm.write(f" Computed log weight for {slp.formatted()}: {log_weight.item()}")
        return LogWeightEstimateFromPrior(log_weight, jnp.array(1, int), jnp.exp(log_weight), 1)
    


dcc_obj = DCCConfig(m, verbose=2,
              init_n_samples=1,
              estimate_weight_n_samples=10_000_000,
              max_iterations=3,
              n_lmh_update_samples=1000,
              max_active_slps=100,
)

t0 = time()

result = dcc_obj.run(jax.random.PRNGKey(0))
result.pprint()

t1 = time()

print(f"Total time: {t1-t0:.3f}s")
comp_time = compilation_time_tracker.get_total_compilation_time_secs()
print(f"Total compilation time: {comp_time:.3f}s ({comp_time / (t1 - t0) * 100:.2f}%)")
