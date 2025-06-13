import sys
sys.path.append(".")

from dccxjax import *
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
    X, lp = m.generate(jax.random.PRNGKey(i))
    print(len(X), lp)

# slp = slp_from_decision_representative(m, X)
# print(len(slp.branching_decisions.decisions))
