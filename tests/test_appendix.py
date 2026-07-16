
from upix.core import *
import upix.distributions as dist
import jax
import jax.numpy as jnp
from upix.core.concretize_tracer import track_decisions, replay_decisions

@model
def mixture(p: float):
    b = sample("b", dist.Bernoulli(p))
    if b:
        sample("a", dist.Normal(1,1))
    else:
        sample("c", dist.Normal(-1,1))

m: Model = mixture(0.4)

key = jax.random.key(0)
dr, lp = m.generate(key) # {"b": Array(1), "a": Array(-0.25747764)}
_, decisions = track_decisions(m.log_prob)(dr)
print(decisions)
slp_log_prob = jax.jit(replay_decisions(m.log_prob, decisions))
print(slp_log_prob(dr)) # (-2.6258543, true)
print(slp_log_prob({"b": jnp.array(0), "a": jnp.array(-1.1)}))
# print(slp_log_prob({"b": jnp.array(0), "c": jnp.array(-1.1)}))

slp = SLP(m, dr, decisions)
print(slp.log_prob({"b": jnp.array(1), "a": jnp.array(1.1)}))  # -1.8402293
print(slp.path_indicator({"b": jnp.array(1), "a": jnp.array(1.1)})) # true
print(slp.path_indicator({"b": jnp.array(0), "c": jnp.array(-1.1)})) # false
print(slp.path_indicator({"b": jnp.array(0), "a": jnp.array(-1.1)})) # false