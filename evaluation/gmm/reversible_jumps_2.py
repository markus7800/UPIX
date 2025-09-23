import jax
import jax.numpy as jnp
from dccxjax.core import *
from dccxjax.infer.involutive import *
import dccxjax.distributions as dist
import numpyro.distributions as numpyro_dists
from typing import NamedTuple


class SplitParams(NamedTuple):
    w1: FloatArray
    mu1: FloatArray
    var1: FloatArray
    w2: FloatArray
    mu2: FloatArray
    var2: FloatArray
    
# for 0...K \ {j_star} returns numbers in ascending order A = 0...K+1, where j1 and j2 are missing
# j != j_start
def split_idx(j, j_star, j1, j2):
    shift1 = -((j > j_star).astype(int))
    shift2 = (j + shift1) >= jnp.minimum(j1, j2)
    shift3 = (j + shift1 + shift2) >= jnp.maximum(j1, j2)
    return j + shift1 + shift2 + shift3
# split_idx(jnp.arange(0,10), 4, 2, 7) = [0, 1, 3, 4, *, 5, 6, 8, 9, 10]
# DiscreteUniform(a,b) has support a...b 

# for 0...K \ {j1,j2} returns numbers in ascending order 0...K-1 with j_star missing
# j != j1 and j != j2
def merge_idx(j, j_star, j1, j2):
    shift1 = -(j > jnp.minimum(j1, j2)).astype(int)
    shift2 = -(j > jnp.maximum(j1, j2)).astype(int)
    shift3 = (j + shift1 + shift2) >= j_star
    return j + shift1 + shift2 + shift3
    
def get_split_params(w, mu, var, u1, u2, u3):
    w1, w2 = w * u1, w * (1 - u1)
    mu1, mu2 = mu - u2 * jnp.sqrt(var * w2/w1), mu + u2 * jnp.sqrt(var * w1/w2)
    var1, var2 = u3 * (1 - jnp.square(u2)) * var * w/w1, (1 - u3) * (1 - jnp.square(u2)) * var * w/w2
    # j1 always gets the smaller of mu1, mu2
    return SplitParams(w1, mu1, var1, w2, mu2, var2)

@model
def aux(model_trace: Trace, ys: jax.Array):
    K = model_trace["K"]
    split = sample("split", dist.Bernoulli(jax.lax.select(K == 0, 1., 0.5)))
    if split == 1:
        split_aux(model_trace, K, ys)
    else:
        merge_aux(model_trace, K, ys)

def split_aux(model_trace: Trace, K: IntArrayLike, ys: jax.Array):
    j_star = sample("j_star", dist.DiscreteUniform(0,K))
    r1 = sample("r1", dist.DiscreteUniform(0,K+1))
    r2 = sample("r2", dist.DiscreteUniform(0,K))
    u1 = sample("u1", dist.Beta(2.,2.))
    t = dist.Transform(numpyro_dists.transforms.AffineTransform(jnp.array([0.,0.],float),jnp.array([1.,-1.],float)))
    td = dist.TransformedDistribution(dist.Beta(jnp.array([2.,2.],float),jnp.array([2.,2.],float)), t)
    md =  dist.MixtureSameFamily(dist.Categorical(jnp.array([0.5,0.5],float)), td)
    u2 = sample("u2", md)
    u3 = sample("u3", dist.Beta(1.,1.))
    
    w = model_trace["w"][j_star]
    mu = model_trace["mus"][j_star]
    var = model_trace["vars"][j_star]
    zs = model_trace["zs"]
    sp = get_split_params(w, mu, var, u1, u2, u3)
    w1, mu1, var1, w2, mu2, var2 = sp.w1, sp.mu1, sp.var1, sp.w2, sp.mu2, sp.var2
    
    log_p = jnp.log(w1) + dist.Normal(mu1, jnp.sqrt(var1)).log_prob(ys)
    log_q = jnp.log(w2) + dist.Normal(mu2, jnp.sqrt(var2)).log_prob(ys)
    
    to_first = sample("to_first", dist.MaskedBernoulli(jnp.exp(log_p - jnp.logaddexp(log_p, log_q)), (zs == j_star)))
    

def split_involution(old_model_trace: Trace, old_aux_trace: Trace, new_model_trace: Trace, new_aux_trace: Trace):
    K = int(branching(old_model_trace["K"]))
    
    r1 = read_discrete(old_aux_trace, "r1")
    r2 = read_discrete(old_aux_trace, "r2")
    j_star = read_discrete(old_aux_trace, "j_star")
    to_first = read_discrete(old_aux_trace, "to_first")
    u1 = read_continuous(old_aux_trace, "u1")
    u2 = read_continuous(old_aux_trace, "u2")
    u3 = read_continuous(old_aux_trace, "u3")
    
    j1, j2 = r1, r2 + (r2 >= r1)
    
    w = read_continuous(old_model_trace, "w")[j_star]
    mu = read_continuous(old_model_trace, "mus")[j_star]
    var = read_continuous(old_model_trace, "vars")[j_star]
    zs = read_discrete(old_model_trace, "zs")
    
    sp = get_split_params(w, mu, var, u1, u2, u3)
    
    w1, mu1, var1, w2, mu2, var2 = sp.w1, sp.mu1, sp.var1, sp.w2, sp.mu2, sp.var2
    
    K = K + 1
    w_new = jnp.zeros((K+1,))
    mus_new = jnp.zeros((K+1,))
    vars_new = jnp.zeros((K+1,))

    ixs = split_idx(jnp.arange(0,K+1-1),j_star,j1,j2)
    w_new = w_new.at[ixs].set(read_continuous(old_model_trace, "w"))
    mus_new = mus_new.at[ixs].set(read_continuous(old_model_trace, "mus"))
    vars_new = vars_new.at[ixs].set(read_continuous(old_model_trace, "vars"))

    w_new = w_new.at[j1].set(w1)
    w_new = w_new.at[j2].set(w2)

    mus_new = mus_new.at[j1].set(mu1)
    mus_new = mus_new.at[j2].set(mu2)

    vars_new = vars_new.at[j1].set(var1)
    vars_new = vars_new.at[j2].set(var2)

    zs_new = split_idx(zs, j_star, j1, j2)
    zs_new = jnp.where((zs == j_star) & to_first, j1, zs_new)
    zs_new = jnp.where((zs == j_star) & ~to_first, j2, zs_new)

    write_discrete(new_model_trace, "K", K)
    write_continuous(new_model_trace, "w", w_new)
    write_continuous(new_model_trace, "mus", mus_new)
    write_continuous(new_model_trace, "vars", vars_new)
    write_discrete(new_model_trace, "zs", zs_new)

    write_discrete(new_aux_trace, "j_star", j_star)
    write_discrete(new_aux_trace, "r1", r1)
    write_discrete(new_aux_trace, "r2", r2)


def merge_aux(model_trace: Trace, K: IntArrayLike, ys: jax.Array):
    sample("j_star", dist.DiscreteUniform(0,K-1))
    sample("r1", dist.DiscreteUniform(0,K))
    sample("r2",dist.DiscreteUniform(0,K-1))
    
def merge_involution(old_model_trace: Trace, old_aux_trace: Trace, new_model_trace: Trace, new_aux_trace: Trace):
    K = int(branching(old_model_trace["K"]))
    
    r1 = read_discrete(old_aux_trace, "r1")
    r2 = read_discrete(old_aux_trace, "r2")
    j_star = read_discrete(old_aux_trace, "j_star")
    
    j1, j2 = r1, r2 + (r2 >= r1)

    ws = read_continuous(old_model_trace, "w")
    mus = read_continuous(old_model_trace, "mus")
    vars = read_continuous(old_model_trace, "vars")
    zs = read_discrete(old_model_trace, "zs")
    
    w1, w2 = ws[j1], ws[j2]
    mu1, mu2 = mus[j1], mus[j2]
    var1, var2 = vars[j1], vars[j2]
    
    w = w1 + w2
    mu = (w1*mu1 + w2*mu2) / w
    var = -jnp.square(mu) + (w1*(jnp.square(mu1) + var1) + w2*(jnp.square(mu2) + var2)) / w

    u1 = w1/w
    u2 = (mu - mu1) / jnp.sqrt(var * w2/w1) # u2 can be smaller than 0 here because we have not guaranteeed mu1 < mu2
    u3 = var1/var * u1 / (1 - jnp.square(u2))


    K = K - 1

    w_new = jnp.zeros((K+1,))
    mus_new = jnp.zeros((K+1,))
    vars_new = jnp.zeros((K+1,))
    
    ixs = merge_idx(jnp.arange(0,K+1+1), j_star, j1, j2)
    w_new = w_new.at[ixs].set(ws)
    mus_new = mus_new.at[ixs].set(mus)
    vars_new = vars_new.at[ixs].set(vars)


    w_new = w_new.at[j_star].set(w)
    mus_new = mus_new.at[j_star].set(mu)
    vars_new = vars_new.at[j_star].set(var)

    zs_new = merge_idx(zs, j_star, j1, j2)
    zs_new = jnp.where((zs == j1) | (zs == j2), j_star, zs_new)
    to_first = (zs == j1).astype(int)

    write_discrete(new_model_trace, "K", K)
    write_continuous(new_model_trace, "w", w_new)
    write_continuous(new_model_trace, "mus", mus_new)
    write_continuous(new_model_trace, "vars", vars_new)
    write_discrete(new_model_trace, "zs", zs_new)
    
    write_discrete(new_aux_trace, "j_star", j_star)
    write_discrete(new_aux_trace, "r1", r1)
    write_discrete(new_aux_trace, "r2", r2)
    write_continuous(new_aux_trace, "u1", u1)
    write_continuous(new_aux_trace, "u2", u2)
    write_continuous(new_aux_trace, "u3", u3)
    write_discrete(new_aux_trace, "to_first", to_first)
    
    
def involution(old_model_trace: Trace, old_aux_trace: Trace, new_model_trace: Trace, new_aux_trace: Trace):
    split = read_discrete(old_aux_trace, "split")
    if split == 1:
        split_involution(old_model_trace, old_aux_trace, new_model_trace, new_aux_trace)
        write_discrete(new_aux_trace, "split", 0)
    else:
        merge_involution(old_model_trace, old_aux_trace, new_model_trace, new_aux_trace)
        write_discrete(new_aux_trace, "split", 1)

        
        
from gmm import *

m = gmm(ys)
m.set_slp_formatter(formatter)
m.set_slp_sort_key(find_K)

# K_test = 3

# model_trace, _ = m.generate(jax.random.key(0), {"K": jnp.array(K_test,int)})
# slp = slp_from_decision_representative(m, model_trace)

# aux_model = aux(model_trace, ys)
# aux_trace, aux_lp = aux_model.generate(jax.random.key(0))

# print("model_trace:", model_trace)
# print("aux_trace:", aux_trace, aux_lp)
# tt = TraceTansformation(involution)

# with tt:
#     new_model_trace, new_aux_trace = tt.apply(model_trace, aux_trace)
#     j = tt.jacobian(model_trace, aux_trace)
#     print(j.shape, jnp.linalg.slogdet(j).logabsdet)
# print("new_model_trace:", new_model_trace)
# print("new_aux_trace:", new_aux_trace)

# # round trip

# with tt:
#     round_trip_model_trace, round_trip_aux_trace = tt.apply(new_model_trace, new_aux_trace)
#     j = tt.jacobian(new_model_trace, new_aux_trace)
#     print(j.shape, jnp.linalg.slogdet(j).logabsdet)

# print("round_trip_model_trace:", round_trip_model_trace)
# print("round_trip_aux_trace:", round_trip_aux_trace)


from dccxjax.core.branching_tracer import trace_branching, retrace_branching

def get_index(trace: Trace):
    return trace["K"]
def rjmcmc_move(m: Model, K: int, ys: jax.Array):
    model_trace, lp = m.generate(jax.random.key(0), {"K": jnp.array(K,int)})
    tt = TraceTansformation(involution)
    
    def _move(model_trace: Trace, lp: FloatArray, key: PRNGKey):
        generate_key, accept_key = jax.random.split(key)
        aux_model = aux(model_trace, ys)
        aux_trace, aux_lp_forward = aux_model.generate(generate_key)
        with tt:
            new_model_trace, new_aux_trace = tt.apply(model_trace, aux_trace)
            j = tt.jacobian(model_trace, aux_trace)
        logabsdetJ = jnp.linalg.slogdet(j).logabsdet
        aux_lp_backward = aux_model.log_prob(new_aux_trace)
        
        log_alpha = m.log_prob(new_model_trace) - lp + aux_lp_backward - aux_lp_forward + logabsdetJ
        accept = jax.lax.log(jax.random.uniform(accept_key)) < log_alpha
        return jax.lax.select(accept, get_index(new_model_trace), get_index(model_trace))

        
    ret, decisions = trace_branching(_move, model_trace, lp, jax.random.key(0))
    print(decisions.to_human_readable())
    jitted_move = jax.jit(retrace_branching(_move, decisions))
    print(jitted_move(model_trace, lp, jax.random.key(0)))
    
rjmcmc_move(m, 3, ys)
