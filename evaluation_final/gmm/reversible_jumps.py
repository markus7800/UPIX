from dccxjax import *
import numpyro.distributions as numpyro_dists
from typing import NamedTuple
from jax.flatten_util import ravel_pytree

class SplitAux(NamedTuple):
    j_star: int
    r1: int
    r2: int
    u1: float
    u2: float
    u3: float

# for 0...K \ {j_star} returns numbers in ascending order A = 0...K+1, where j1 and j2 are missing
# j != j_start
def split_idx(j, j_star, j1, j2):
    shift1 = -((j > j_star).astype(int))
    shift2 = (j + shift1) >= min(j1, j2)
    shift3 = (j + shift1 + shift2) >= max(j1, j2)
    return j + shift1 + shift2 + shift3
# split_idx(jnp.arange(0,10), 4, 2, 7) = [0, 1, 3, 4, *, 5, 6, 8, 9, 10]
# DiscreteUniform(a,b) has support a...b 

def split_randomness(rng_key: PRNGKey, K: int):
    j_key, r1_key, r2_key, u1_key, u2_key, u3_key = jax.random.split(rng_key, 6)

    j_star = numpyro_dists.DiscreteUniform(0,K).sample(j_key)

    r1 = numpyro_dists.DiscreteUniform(0,K+1).sample(r1_key)

    r2 = numpyro_dists.DiscreteUniform(0,K).sample(r2_key)

    u1 = numpyro_dists.Beta(2.,2.).sample(u1_key)

    u2 = numpyro_dists.Beta(2.,2.).sample(u2_key)

    u3 = numpyro_dists.Beta(1.,1.).sample(u3_key)

    return SplitAux(j_star, r1, r2, u1, u2, u3) # type: ignore

def split_randomness_logQ(aux: SplitAux, K: int):
    logQ = 0.

    logQ += numpyro_dists.DiscreteUniform(0,K).log_prob(aux.j_star)

    logQ += numpyro_dists.DiscreteUniform(0,K+1).log_prob(aux.r1)

    logQ += numpyro_dists.DiscreteUniform(0,K).log_prob(aux.r2)

    logQ += numpyro_dists.Beta(2.,2.).log_prob(aux.u1)

    logQ += numpyro_dists.Beta(2.,2.).log_prob(aux.u2)

    logQ += numpyro_dists.Beta(2.,2.).log_prob(aux.u3)

    return logQ

class MergeAux(NamedTuple):
    j_star: int
    r1: int
    r2: int

# for 0...K \ {j1,j2} returns numbers in ascending order 0...K-1 with j_star missing
# j != j1 and j != j2
def merge_idx(j, j_star, j1, j2):
    shift1 = -(j > min(j1, j2)).astype(int)
    shift2 = -(j > max(j1, j2)).astype(int)
    shift3 = (j + shift1 + shift2) >= j_star
    return j + shift1 + shift2 + shift3
# merge_idx(jnp.arange(0,10), 4, 2, 7) = Array([0, 1, *, 2, 3, 5, 6, *, 7, 8], dtype=int32)
def merge_randomness(rng_key: PRNGKey, K: int):
    j_key, r1_key, r2_key = jax.random.split(rng_key, 3)

    j_star = numpyro_dists.DiscreteUniform(0,K-1).sample(j_key)

    r1 = numpyro_dists.DiscreteUniform(0,K).sample(r1_key)

    r2 = numpyro_dists.DiscreteUniform(0,K-1).sample(r2_key)

    return MergeAux(j_star, r1, r2) # type: ignore

def merge_randomness_logQ(aux: MergeAux, K: int):
    logQ = 0.

    logQ += numpyro_dists.DiscreteUniform(0,K-1).log_prob(aux.j_star)

    logQ += numpyro_dists.DiscreteUniform(0,K).log_prob(aux.r1)

    logQ += numpyro_dists.DiscreteUniform(0,K-1).log_prob(aux.r2)

    return logQ


def compute_abs_det_J_split(w, mu, var, u1, u2, u3, w1, w2, mu1, mu2, var1, var2):
    return (w * jnp.abs(mu1 - mu2) * var1 * var2) / (u2 * (1 - jnp.square(u2)) * u3 * (1 - jnp.square(u3)) * var)

def compute_log_abs_det_J_split(w, mu, var, u1, u2, u3, w1, w2, mu1, mu2, var1, var2):
    return jnp.log(w) + jnp.log(jnp.abs(mu1 - mu2)) + jnp.log(var1) + jnp.log(var2) - jnp.log(u2) - jnp.log(1 - jnp.square(u2)) - jnp.log(u3) - jnp.log(1 - jnp.square(u3)) - jnp.log(var)

def split_move(X: Trace, lp: float, rng_key: PRNGKey, K: int, model_log_prob: Callable[[Trace],FloatArray], check=False):
    aux_key, accept_key = jax.random.split(rng_key, 2)
    split_aux = split_randomness(aux_key, K)
    print(f"{X=}")
    print(f"{split_aux=}")
    logQ_split = split_randomness_logQ(split_aux, K)

    X_new, merge_aux, logabsdetJ = split_involution(X, split_aux, K)
    print(f"{X_new=} {K+1=}")

    if check:
        X_2, split_aux_2, logabsdetJ_2 = merge_involution(X_new, merge_aux, K+1)
        print(f"{X_2=}")
        print(f"{split_aux_2=}")
        for addr, value in X_2.items():
            assert jnp.all(jnp.isclose(value, X[addr], rtol=0, atol=1e-3)), f"{value} vs {X[addr]}"
        split_aux_2_flat, _ = ravel_pytree(split_aux_2)
        split_aux_flat, _ = ravel_pytree(split_aux)
        assert jnp.all(jnp.isclose(split_aux_2_flat, split_aux_flat, rtol=0, atol=1e-3))
        # assert jnp.isclose(logabsdetJ_2, 1-logabsdetJ, rtol=0, atol=1e-3), f"{logabsdetJ_2} vs {1-logabsdetJ}"
        print("Check: ok.")

    # logQ_merge = merge_randomness_logQ(merge_aux, K)

    # log_alpha = model_log_prob(X_new) - lp + logQ_merge - logQ_split + detJ

    # # aux merge trace = (j_star, r1, r2)
    # return jax.random.uniform(accept_key) < log_alpha


def split_involution(X: Trace, aux: SplitAux, K: int):
    j_star, r1, r2, u1, u2, u3 = aux.j_star, aux.r1, aux.r2, aux.u1, aux.u2, aux.u3

    j1, j2 = r1, r2 + (r2 >= r1)

    w = X["w"][j_star]
    mu = X["mus"][j_star]
    var = X["vars"][j_star]

    w1, w2 = w * u1, w * (1 - u1)
    mu1, mu2 = mu - u2 * jnp.sqrt(var * w2/w1), mu + u2 * jnp.sqrt(var * w1/w2)
    var1, var2 = u3 * (1 - jnp.square(u2)) * var * w/w1, (1 - u3) * (1 - jnp.square(u2)) * var * w/w2

    logabsdetJ = compute_log_abs_det_J_split(w, mu, var, u1, u2, u3, w1, w2, mu1, mu2, var1, var2)

    K = K + 1
    w_new = jnp.zeros((K+1,))
    mus_new = jnp.zeros((K+1,))
    vars_new = jnp.zeros((K+1,))

    ixs = [str(ix) if i != j_star else "*" for i, ix in enumerate(split_idx(jnp.arange(0,K+1-1), j_star, j1, j2))]
    print(f"split ixs {j_star=} {j1=} {j2=} {ixs} {w_new.shape=} {X['w'].shape=} {K=}") # w_new.shape = (K+1,) X['w'].shape == (K,)
    ixs1 = split_idx(jnp.arange(0,j_star),j_star,j1,j2)
    ixs2 = split_idx(jnp.arange(j_star+1,K+1-1),j_star,j1,j2)

    print(f"{ixs1=} {ixs2=}")
    ixs1_ = list(range(0,j_star))
    ixs2_ = list(range(j_star+1,K+1-1))

    print(f"range(0,j_star)={ixs1_} range(j_star+1,K+1-1)={ixs2_}")

    w_new = w_new.at[ixs1].set(X["w"][:j_star])
    mus_new = mus_new.at[ixs1].set(X["mus"][:j_star])
    vars_new = vars_new.at[ixs1].set(X["vars"][:j_star])

    w_new = w_new.at[ixs2].set(X["w"][j_star+1:])
    mus_new = mus_new.at[ixs2].set(X["mus"][j_star+1:])
    vars_new = vars_new.at[ixs2].set(X["vars"][j_star+1:])

    w_new = w_new.at[j1].set(w1)
    w_new = w_new.at[j2].set(w2)

    mus_new = mus_new.at[j1].set(mu1)
    mus_new = mus_new.at[j2].set(mu2)

    vars_new = vars_new.at[j1].set(var1)
    vars_new = vars_new.at[j2].set(var2)

    # TODO: zs


    X_new = {"K": K, "w": w_new, "mus": mus_new, "vars": vars_new}
    aux_new = MergeAux(j_star, r1, r2)

    return X_new, aux_new, logabsdetJ

def merge_move(X: Trace, lp: float, rng_key: PRNGKey, K: int, model_log_prob: Callable[[Trace],FloatArray], check=False):
    aux_key, accept_key = jax.random.split(rng_key, 2)
    merge_aux = merge_randomness(aux_key, K)
    logQ_merge = merge_randomness_logQ(merge_aux, K)

    X_new, split_aux, logabsdetJ = merge_involution(X, merge_aux, K)

    if check:
        X_2, merge_aux_2, logabsdetJ_2 = split_involution(X_new, split_aux, K-1)
        for addr, value in X_2.items():
            assert jnp.all(jnp.isclose(value, X[addr], rtol=0, atol=1e-3)), f"{value} vs {X[addr]}"
        merge_aux_2_flat, _ = ravel_pytree(merge_aux_2)
        merge_aux_flat, _ = ravel_pytree(merge_aux)
        assert jnp.all(jnp.isclose(merge_aux_2_flat, merge_aux_flat, rtol=0, atol=1e-3))
        # assert jnp.isclose(logabsdetJ_2, 1-logabsdetJ, rtol=0, atol=1e-3), f"{logabsdetJ_2} vs {1-logabsdetJ}"
        print("Check: ok.")

    # logQ_split = split_randomness_logQ(split_aux, K)

    # log_alpha = model_log_prob(X_new) - lp + logQ_split - logQ_merge + detJ

    # return jax.random.uniform(accept_key) < log_alpha


def merge_involution(X: Trace, aux: MergeAux, K: int):
    j_star, r1, r2 = aux.j_star, aux.r1, aux.r2

    j1, j2 = r1, r2 + (r2 >= r1)

    w1, w2 = X["w"][j1], X["w"][j2]
    mu1, mu2 = X["mus"][j1], X["mus"][j2]
    var1, var2 = X["vars"][j1], X["vars"][j2]


    w = w1 + w2
    mu = (w1*mu1 + w2*mu2) / w
    var = -jnp.square(mu) + (w1*(jnp.square(mu1) + var1) + w2*(jnp.square(mu2) + var2)) / w

    u1 = w1/w
    u2 = (mu - mu1) / jnp.sqrt(var * w2/w1)
    u3 = var1/var * u1 / (1 - jnp.square(u2))

    logabsdetJ = 1 - compute_log_abs_det_J_split(w, mu, var, u1, u2, u3, w1, w2, mu1, mu2, var1, var2)

    K = K - 1
    min_j = min(j1,j2)
    max_j = max(j1,j2)

    w_new = jnp.zeros((K+1,))
    mus_new = jnp.zeros((K+1,))
    vars_new = jnp.zeros((K+1,))

    ixs = [str(ix) if i not in (j1,j2) else "*" for i, ix in enumerate(merge_idx(jnp.arange(0,K+1+1), j_star, j1, j2))]
    print(f"merge ixs {j_star=} {min_j=} {max_j=} {ixs} {w_new.shape=} {X['w'].shape=} {K=}") # w_new.shape = (K+1,) X['w'].shape == (K+1+1,)

    ixs1 = merge_idx(jnp.arange(0, min_j), j_star, j1, j2)
    ixs3 = merge_idx(jnp.arange(max_j+1,K+1+1), j_star, j1, j2)
    ixs2 = merge_idx(jnp.arange(min_j+1,max_j), j_star, j1, j2)
    print(f"{ixs1=} {ixs2=} {ixs3=}")
    ixs1_ = list(range(0,min_j))
    ixs2_ = list(range(min_j+1,max_j))
    ixs3_ = list(range(max_j+1, K+1+1))

    print(f"range(0,min_j)={ixs1_} range(min_j+1,max_j)={ixs2_} range(max_j+1, K+1+1)={ixs3_}")

    w_new = w_new.at[ixs1].set(X["w"][:min_j])
    mus_new = mus_new.at[ixs1].set(X["mus"][:min_j])
    vars_new = vars_new.at[ixs1].set(X["vars"][:min_j])

    w_new = w_new.at[ixs2].set(X["w"][min_j+1:max_j])
    mus_new = mus_new.at[ixs2].set(X["mus"][min_j+1:max_j])
    vars_new = vars_new.at[ixs2].set(X["vars"][min_j+1:max_j])

    w_new = w_new.at[ixs3].set(X["w"][max_j+1:K+1+1])
    mus_new = mus_new.at[ixs3].set(X["mus"][max_j+1:K+1+1])
    vars_new = vars_new.at[ixs3].set(X["vars"][max_j+1:K+1+1])


    w_new = w_new.at[j_star].set(w)
    mus_new = mus_new.at[j_star].set(mu)
    vars_new = vars_new.at[j_star].set(var)


    # TODO: zs

    X_new = {"K": K, "w": w_new, "mus": mus_new, "vars": vars_new}
    aux_new = SplitAux(j_star, r1, r2, u1, u2, u3) # type: ignore

    return X_new, aux_new, logabsdetJ
