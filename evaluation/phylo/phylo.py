from dccxjax.all import *
import dccxjax.distributions as dist
import jax
from typing import List

from trees import *
from analytical import *

import matplotlib.pyplot as plt

# tree = read_phyjson("evaluation/phylo/data/bisse_32.phyjson")
# rho = jnp.array(1.0,float)

tree = read_phyjson("evaluation/phylo/data/Alcedinidae.phyjson")
# rho = jnp.array(0.5684210526315789,float)
rho = jnp.array(1.0,float)

# miking-benchmarks/benchmark-suite/benchmarks/ppl/phyl/webppl/phywppl/examples/crbd.wppl
# miking-benchmarks/benchmark-suite/benchmarks/ppl/phyl/webppl/phywppl/models/crbd.wppl


# ported from
# miking-benchmarks/benchmark-suite/benchmarks/ppl/phyl/coreppl/crbd.mc
# miking-benchmarks/benchmark-suite/benchmarks/ppl/phyl/rootppl/phyrppl/models/CRBD.cuh

class Counter:
    def __init__(self) -> None:
        self.count = 0
    def inc(self) -> int:
        self.count += 1
        return self.count
    
    
# miking-benchmarks/benchmark-suite/benchmarks/ppl/phyl/pyro/models/crbd.py
def CRBD_goes_undetected_2(c: Counter, start_time: FloatArray, lam: jax.Array, mu: jax.Array, rho: jax.Array) -> bool:
    # start_time == t
    duration = sample(f"duration_{c.inc()}", dist.Exponential(mu)) # == (t - t_end)
    if duration > start_time: # t_end = start_time - duration <= 0
        b = sample(f"b_{c.inc()}", dist.Bernoulli(rho))
        if b == 1:
            return False
    # t_end = max(t_end, 0)
    branch_length = jax.lax.min(duration, start_time) # == (t - t_end)
    n = sample(f"n_{c.inc()}", dist.Poisson(lam * branch_length))
    for _ in range(n):
        event_time = sample(f"event_time_{c.inc()}", dist.Uniform(start_time - branch_length, start_time))
        if not CRBD_goes_undetected_2(c, event_time, lam, mu, rho):
            return False
    return True
def simBranch2(c: Counter, n: IntArray, start_time: FloatArray, stop_time: FloatArray, lam: jax.Array, mu: jax.Array, rho: jax.Array) -> FloatArray:
    f = jnp.array(0.,float)
    for _ in range(n):
        current_time = sample(f"current_time_{c.inc()}", dist.Uniform(stop_time, start_time))
        if CRBD_goes_undetected_2(c, current_time, lam, mu, rho):
            f += jnp.log(2)
        else:
            return jnp.array(-jnp.inf,float)
    return f  

from typing import NamedTuple
class WhileState(NamedTuple):
    mu: jax.Array
    lam: jax.Array
    rho: jax.Array
    stack_ptr: int
    rng_key: PRNGKey
    ns: IntArray
    start_times: FloatArray
    branch_lengths: IntArray
    rets: IntArray
    
def CRBD_goes_undetected_3_body(state: WhileState) -> WhileState:
    def _continue(_s: WhileState):
        return _s.rng_key, _s.ns, _s.start_times, _s.branch_lengths, _s.rets
    def _start(_s: WhileState):
        rng_key, sample_key_1, sample_key_2, sample_key_3 = jax.random.split(_s.rng_key,4)
        
        exp_dist = dist.Exponential(_s.mu)
        duration = exp_dist.sample(sample_key_1)
        branch_length = jax.lax.min(duration, _s.start_times[_s.stack_ptr])
        
        bern_dist = dist.Bernoulli(_s.rho)
        b = bern_dist.sample(sample_key_2)
        
        pois_dist = dist.Poisson(_s.lam * branch_length)
        _n = pois_dist.sample(sample_key_3)
        
        r = (duration > _s.start_times[_s.stack_ptr]) & (b == 1)
        n = jax.lax.select(r, 0, _n) # n == 0 indicates return
        rets = _s.rets.at[_s.stack_ptr].set(r)
        
        return rng_key, _s.ns.at[_s.stack_ptr].set(n), _s.start_times, _s.branch_lengths.at[_s.stack_ptr].set(branch_length), rets
            
    rng_key, ns, start_times, branch_lengths, rets = jax.lax.cond(state.ns[state.stack_ptr] == -1, _start, _continue, operand = state)
    
    def _pop_stack(rng_key, ns, start_times, branch_lengths, rets) -> WhileState:
        rets = rets.at[state.stack_ptr - 1].set(rets[state.stack_ptr]) # pass return value up the stack
        return WhileState(
            state.mu, state.lam, state.rho,
            state.stack_ptr - 1, rng_key, ns,
            state.start_times, state.branch_lengths, rets)
    
    def _dec_n_and_push_stack(rng_key, ns, start_times, branch_lengths, rets) -> WhileState:
        start_time = start_times[state.stack_ptr]
        branch_length = branch_lengths[state.stack_ptr]
        d = dist.Uniform(start_time - branch_length, start_time)
        rng_key, sample_key = jax.random.split(rng_key)
        event_time = d.sample(sample_key)
        start_times = state.start_times.at[state.stack_ptr + 1].set(event_time)
        rets = state.rets.at[state.stack_ptr + 1].set(0)
        ns = ns.at[state.stack_ptr + 1].set(-1)
        
        ns = ns.at[state.stack_ptr].set(ns[state.stack_ptr] - 1) # decrement n
    
        return WhileState(
            state.mu, state.lam, state.rho,
            state.stack_ptr + 1, rng_key, ns,
            start_times, branch_lengths, rets
        )
    
    return jax.lax.cond(
        (ns[state.stack_ptr] == 0),
        _pop_stack, _dec_n_and_push_stack,
        rng_key, ns, start_times, branch_lengths, rets
    )
    


# iter
def walk(c: Counter, n: IntArray, start_time: FloatArray, branch_length: FloatArray, lam: jax.Array, mu: jax.Array, rho: jax.Array) -> bool:
    if (n == 0):
        return True
    else:
        event_time = sample(f"event_time_{c.inc()}", dist.Uniform(start_time - branch_length, start_time))
        if CRBD_goes_undetected(c, event_time, lam, mu, rho):
            return walk(c, n - 1, start_time, branch_length, lam, mu, rho)
        else:
            return False

def CRBD_goes_undetected(c: Counter, start_time: FloatArray, lam: jax.Array, mu: jax.Array, rho: jax.Array) -> bool:
    duration = sample(f"duration_{c.inc()}", dist.Exponential(mu))
    if duration > start_time:
        b = sample(f"b_{c.inc()}", dist.Bernoulli(rho))
        if b == 1:
            return False
    
    branch_length = jax.lax.min(duration, start_time)
    n = sample(f"n_{c.inc()}", dist.Poisson(lam * branch_length))
    return walk(c, n, start_time, branch_length, lam, mu, rho)
        

def simBranch(c: Counter, n: IntArray, start_time: FloatArray, stop_time: FloatArray, lam: jax.Array, mu: jax.Array, rho: jax.Array) -> FloatArray:
    if (n == 0):
        return jnp.array(0.0,float)
    else:
        current_time = sample(f"current_time_{c.inc()}", dist.Uniform(stop_time, start_time))
        if CRBD_goes_undetected(c, current_time, lam, mu, rho):
            v = simBranch(c, n-1, start_time, stop_time, lam, mu, rho)
            return v + jnp.log(2)
        else:
            return jnp.array(-jnp.inf,float)
        

def simTree(c: Counter, tree: Node, parent: Node, lam: jax.Array, mu: jax.Array, rho: jax.Array):
    lnProb1 = -mu * (parent.age - tree.age)
    lnProb2 = jnp.log(lam) if isinstance(tree, Branch) else jnp.log(rho)
    
    start_time = jnp.array(parent.age,float)
    stop_time = jnp.array(tree.age,float)
    n = sample(f"n_{c.inc()}", dist.Poisson(lam * (start_time - stop_time)))
    lnProb3 = simBranch(c, n, start_time, stop_time, lam, mu, rho)
    logfactor(lnProb1 + lnProb2 + lnProb3)
    # resample
    
    if isinstance(tree, Branch):
        simTree(c, tree.left, tree, lam, mu, rho)
        simTree(c, tree.right, tree, lam, mu, rho)

@model
def phylo(tree: Branch, rho: jax.Array):
    lam = sample("lambda", dist.Gamma(1.0, 1.0))
    mu = sample("mu", dist.Gamma(1.0, 0.5))
    # lam = 0.3
    # mu = 0.05
    
    num_leaves = count_leaves(tree)
    corr_factor = (num_leaves - 1) * jnp.log(2.0) - ln_factorial(num_leaves)
    logfactor(corr_factor)
    # resample
    
    c = Counter()
    simTree(c, tree.left, tree, lam, mu, rho)
    simTree(c, tree.right, tree, lam, mu, rho)
    
def f(node: Node):
    if isinstance(node, Branch):
        assert node.age >= node.left.age
        assert node.age >= node.right.age
        f(node.left)
        f(node.right)
f(tree)
    
m = phylo(tree, rho)

for i in range(10):
    X, lp = m.generate(jax.random.key(i))
    print(len(X), lp)

# slp = slp_from_decision_representative(m, X)
# print(len(slp.branching_decisions.decisions))
