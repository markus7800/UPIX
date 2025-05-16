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
from dccxjax.infer.dcc2 import *
from dccxjax.types import _unstack_sample_data

from gibbs_proposals import *
from reversible_jumps import *

import logging
setup_logging(logging.WARNING)

compilation_time_tracker = CompilationTimeTracker()
jax.monitoring.register_event_duration_secs_listener(compilation_time_tracker)


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

@model
def gmm(ys: jax.Array):
    N = ys.shape[0]
    K = sample("K", dist.Poisson(lam-1)) + 1
    w = sample("w", dist.Dirichlet(jnp.full((K,), delta)))
    mus = sample("mus", dist.Normal(jnp.full((K,), xi), jnp.full((K,), 1/jax.lax.sqrt(kappa))))
    vars = sample("vars", dist.InverseGamma(jnp.full((K,), alpha), jnp.full((K,), beta)))
    zs = sample("zs", dist.Categorical(jax.lax.broadcast(w, (N,))))
    sample("ys", dist.Normal(mus[zs], jax.lax.sqrt(vars[zs])), observed=ys)


m = gmm(ys)

def find_K(slp: SLP) -> int:
    return int(slp.decision_representative["K"].item())
def formatter(slp: SLP):
    K = find_K(slp) + 1
    return f"#clusters={K}"
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_K)

@dataclass
class RJMCMCTransitionProbEstimate(LogWeightEstimate):
    transition_log_probs: Dict[Any, FloatArray]
    n_samples: int
    def combine_estimate(self, other: LogWeightEstimate) -> "RJMCMCTransitionProbEstimate":
        assert isinstance(other, RJMCMCTransitionProbEstimate)

        n_combined_samples = self.n_samples + other.n_samples
        a = self.n_samples / n_combined_samples
        new_transition_log_probs: Dict[Any, FloatArray] = dict()

        for key in self.transition_log_probs.keys() | other.transition_log_probs.keys():
            est1 = self.transition_log_probs.get(key, -jnp.inf)
            est2 = other.transition_log_probs.get(key, -jnp.inf)
            new_transition_log_probs[key] = jnp.logaddexp(est1 + jax.lax.log(a), est2 + jax.lax.log(1 - a))

        return RJMCMCTransitionProbEstimate(new_transition_log_probs, n_combined_samples)

class DCCConfig(MCMCDCC[DCC_COLLECT_TYPE]):
    def get_MCMC_inference_regime(self, slp: SLP) -> MCMCRegime:
        return MCMCSteps(
            MCMCStep(SingleVariable("w"), MH(WProposal(delta, slp.decision_representative["K"].item()))),
            MCMCStep(SingleVariable("mus"), MH(MusProposal(ys, kappa, xi, slp.decision_representative["K"].item()))),
            MCMCStep(SingleVariable("vars"), MH(VarsProposal(ys, alpha, beta, slp.decision_representative["K"].item()))),
            MCMCStep(SingleVariable("zs"), MH(ZsProposal(ys))),
        )
    
    def initialise_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], rng_key: jax.Array):
        rng_key, generate_key = jax.random.split(rng_key)
        trace, _ = self.model.generate(generate_key, {"K": jnp.array(0,int)})
        slp = slp_from_decision_representative(self.model, trace)
        active_slps.append(slp)
        tqdm.write(f"Make SLP {slp.formatted()} active.")

    def update_active_slps(self, active_slps: List[SLP], inactive_slps: List[SLP], inference_results: Dict[SLP, List[InferenceResult]], log_weight_estimates: Dict[SLP, List[LogWeightEstimate]], rng_key: PRNGKey):
        last_slp = active_slps.pop()
        inactive_slps.append(last_slp)
        estimates = self.get_log_weight_estimates(last_slp)
        estimate = estimates[0] # we only have one per slp
        assert isinstance(estimate, RJMCMCTransitionProbEstimate)
        K = find_K(last_slp)
        split_transition_log_prob = estimate.transition_log_probs[(K, K+1)]
        if split_transition_log_prob > jnp.log(0.01):
            trace, _ = self.model.generate(rng_key, {"K": jnp.array(K+1,int)})
            slp = slp_from_decision_representative(self.model, trace)
            active_slps.append(slp)
            tqdm.write(f"Make SLP {slp.formatted()} active.")

    def estimate_log_weight(self, slp: SLP, rng_key: jax.Array) -> RJMCMCTransitionProbEstimate:
        last_inference_result = self.inference_results[slp][-1]
        assert isinstance(last_inference_result, MCMCInferenceResult)
        assert not self.config.get("mcmc_optimise_memory_with_early_return_map", False)
        traces: Trace = last_inference_result.value_tree[0]
        lps: FloatArray = last_inference_result.value_tree[1]
        traces = jax.tree_util.tree_map(_unstack_sample_data, traces)
        lps = _unstack_sample_data(lps)

        K = find_K(slp)

        def split(X: Trace, lp: FloatArray, rng_key: PRNGKey):
            return split_move(X, lp, rng_key, K, ys, self.model.log_prob)
        
        def merge(X: Trace, lp: FloatArray, rng_key: PRNGKey):
            return merge_move(X, lp, rng_key, K, ys, self.model.log_prob)
        
        transition_log_probs: Dict[Any, FloatArray] = dict()
        n_samples = lps.size
        split_transition_log_prob = jax.scipy.special.logsumexp(jax.jit(jax.vmap(split))(traces, lps, jax.random.split(rng_key, lps.shape[0]))) - jnp.log(n_samples)
        if K == 0: 
            transition_log_probs[(K, K+1)] = split_transition_log_prob
            tqdm.write(f"Estimate log weight for {slp.formatted()}: split prob = {jnp.exp(split_transition_log_prob.item()):.6f}")
        else:
            merge_transition_log_prob = jax.scipy.special.logsumexp(jax.jit(jax.vmap(merge))(traces, lps, jax.random.split(rng_key, lps.shape[0]))) - jnp.log(n_samples)

            merge_transition_log_prob = merge_transition_log_prob - jnp.log(2)
            split_transition_log_prob = split_transition_log_prob - jnp.log(2)

            transition_log_probs[(K, K+1)] = split_transition_log_prob
            transition_log_probs[(K, K-1)] = merge_transition_log_prob
            tqdm.write(f"Estimate log weight for {slp.formatted()}: split prob = {jnp.exp(split_transition_log_prob).item():.6f} merge prob = {jnp.exp(merge_transition_log_prob).item():.6f}")

        return RJMCMCTransitionProbEstimate(transition_log_probs, n_samples)

    
    def compute_slp_log_weight(self, log_weight_estimates: Dict[SLP, LogWeightEstimate]) -> Dict[SLP, FloatArray]:
        max_K = max(find_K(slp) for slp in log_weight_estimates.keys())
        visited_k = jnp.zeros((max_K+1,))
        bayes_factor = jnp.zeros((max_K+1, max_K+1))
        for estimate in log_weight_estimates.values():
            assert isinstance(estimate, RJMCMCTransitionProbEstimate)
            for (current_k, next_k), log_prob in estimate.transition_log_probs.items():
                visited_k = visited_k.at[current_k].set(1)
                if next_k <= max_K:
                    bayes_factor = bayes_factor.at[current_k, next_k].add(log_prob)
                    bayes_factor = bayes_factor.at[next_k, current_k].add(-log_prob)
        assert jnp.all(visited_k == 1)
        for i in range(max_K+1):
            for j in range(i+1,max_K+1):
                bayes_factor = bayes_factor.at[i, j].set(bayes_factor[i,j-1] - bayes_factor[j, j-1])
                bayes_factor = bayes_factor.at[j, i].set(-bayes_factor[i,j])
        
        k_to_log_weight = -jax.scipy.special.logsumexp(bayes_factor, axis=1)

        result: Dict[SLP, FloatArray] = dict()
        for slp in log_weight_estimates.keys():
            k = find_K(slp)
            result[slp] = k_to_log_weight[k]

        return result

        


dcc_obj = DCCConfig(m, verbose=2,
              mcmc_n_chains=10,
              mcmc_n_samples_per_chain=25_000,
              mcmc_collect_for_all_traces=True,
              estimate_weight_n_samples=1000)

# takes ~185s for 10 * 25_000 * 11 samples
t0 = time()

result = dcc_obj.run(jax.random.PRNGKey(0))
result.pprint()

t1 = time()

print(f"Total time: {t1-t0:.3f}s")
comp_time = compilation_time_tracker.get_total_compilation_time_secs()
print(f"Total compilation time: {comp_time:.3f}s ({comp_time / (t1 - t0) * 100:.2f}%)")