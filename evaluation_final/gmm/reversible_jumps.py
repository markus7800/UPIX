from dccxjax import *
import numpyro.distributions as dist

# for 0...K \ j_star returns numbers in ascending order A = 0...K+1, where j1 and j2 are missing
# j != j_start
def split_idx(j, j_star, j1, j2):
    shift1 = -((j > j_star).astype(int))
    shift2 = (j + shift1) >= min(j1, j2)
    shift3 = (j + shift1 + shift2) >= max(j1, j2)
    return j + shift1 + shift2 + shift3

# DiscreteUniform(a,b) has support a...b 

def split_randomness(rng_key: PRNGKey, K: int):
    j_key, r1_key, r2_key, u1_key, u2_key, u3_key = jax.random.split(rng_key, 6)

    j_star = dist.DiscreteUniform(0,K).sample(j_key)

    r1 = dist.DiscreteUniform(0,K+1).sample(r1_key)

    r2 = dist.DiscreteUniform(0,K).sample(r2_key)

    u1 = dist.Beta(2.,2.).sample(u1_key)

    u2 = dist.Beta(2.,2.).sample(u2_key)

    u3 = dist.Beta(1.,1.).sample(u3_key)

    return j_star, r1, r2, u1, u2, u3

def split_randomness_logQ(j_star, r1, r2, u1, u2, u3, K: int):
    logQ = 0.

    logQ += dist.DiscreteUniform(0,K).log_prob(j_star)

    logQ += dist.DiscreteUniform(0,K+1).log_prob(r1)

    logQ += dist.DiscreteUniform(0,K).log_prob(r2)

    logQ += dist.Beta(2.,2.).log_prob(u1)

    logQ += dist.Beta(2.,2.).log_prob(u2)

    logQ += dist.Beta(2.,2.).log_prob(u3)

    return logQ

# j != j1 and j != j2
def merge_idx(j, j_star, j1, j2):
    shift1 = -(j > min(j1, j2)).astype(int)
    shift2 = -(j > max(j1, j2)).astype(int)
    shift3 = (j + shift1 + shift2) >= j_star
    return j + shift1 + shift2 + shift3

def merge_randomness(rng_key: PRNGKey, K: int):
    j_key, r1_key, r2_key = jax.random.split(rng_key, 3)

    j_star = dist.DiscreteUniform(0,K-1).sample(j_key)

    r1 = dist.DiscreteUniform(0,K).sample(r1_key)

    r2 = dist.DiscreteUniform(0,K-1).sample(r2_key)

    return j_star, r1, r2

def merge_randomness_logQ(j_star, r1, r2, K: int):
    logQ = 0.

    logQ += dist.DiscreteUniform(0,K-1).log_prob(j_star)

    logQ += dist.DiscreteUniform(0,K).log_prob(r1)

    logQ += dist.DiscreteUniform(0,K-1).log_prob(r2)

    return logQ


def compute_det_J_split(w, mu, var, u1, u2, u3, w1, w2, mu1, mu2, var1, var2):
    return (w * jnp.abs(mu1 - mu2) * var1 * var2) / (u2 * (1 - jnp.square(u2)) * u3 * (1 - jnp.square(u3)) * var)

def split_move(X: Trace, lp: float, rng_key: PRNGKey, K: int, model_log_prob: Callable[[Trace],FloatArray]):
    aux_key, accept_key = jax.random.split(rng_key, 2)
    j_star, r1, r2, u1, u2, u3 = split_randomness(aux_key, K)
    logQ_split = split_randomness_logQ(j_star, r1, r2, u1, u2, u3, K)

    j1, j2 = r1, r2 + (r2 >= r1)

    w = X["w"][j_star]
    mu = X["mus"][j_star]
    var = X["vars"][j_star]

    w1, w2 = w * u1, w * (1 - u1)
    mu1, mu2 = mu - u2 * jnp.sqrt(var * w2/w1), mu + u2 * jnp.sqrt(var * w1/w2)
    var1, var2 = u3 * (1 - u2^2) * var * w/w1, (1 - u3) * (1 - u2^2) * var * w/w2

    detJ = compute_det_J_split(w, mu, var, u1, u2, u3, w1, w2, mu1, mu2, var1, var2)

    K = K + 1
    w_new = jnp.zeros((K,))
    w_new = w_new.at[split_idx(jnp.arange(0,j_star),j_star,j1,j2)].set(X["w"][:j_star])
    w_new = w_new.at[split_idx(jnp.arange(j_star+1,K-1),j_star,j1,j2)].set(X["w"][j_star+1:])
    w_new = w_new.at[j1].set(w1)
    w_new = w_new.at[j2].set(w2)

    mus_new = jnp.zeros((K,))
    mus_new = mus_new.at[split_idx(jnp.arange(0,j_star),j_star,j1,j2)].set(X["mus"][:j_star])
    mus_new = mus_new.at[split_idx(jnp.arange(j_star+1,K-1),j_star,j1,j2)].set(X["mus"][j_star+1:])
    mus_new = mus_new.at[j1].set(mu1)
    mus_new = mus_new.at[j2].set(mu2)

    vars_new = jnp.zeros((K,))
    vars_new = vars_new.at[split_idx(jnp.arange(0,j_star),j_star,j1,j2)].set(X["vars"][:j_star])
    vars_new = vars_new.at[split_idx(jnp.arange(j_star+1,K-1),j_star,j1,j2)].set(X["vars"][j_star+1:])
    vars_new = vars_new.at[j1].set(var1)
    vars_new = vars_new.at[j2].set(var2)

    # TODO: do zs

    # aux merge trace = (j_star, r1, r2)
    logQ_merge = merge_randomness_logQ(j_star, r1, r2, K)

    X_new = {"k": K, "w": w_new, "mus": mus_new, "vars": vars_new}

    log_alpha = model_log_prob(X_new) - lp + logQ_merge - logQ_split + detJ

    return jax.random.uniform(accept_key) < log_alpha


def merge_move(X: Trace, lp: float, rng_key: PRNGKey, K: int, model_log_prob: Callable[[Trace],FloatArray]):
    aux_key, accept_key = jax.random.split(rng_key, 2)
    j_star, r1, r2 = merge_randomness(aux_key, K)
    logQ_merge = merge_randomness_logQ(j_star, r1, r2, K)

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

    detJ = 1 / compute_det_J_split(w, mu, var, u1, u2, u3, w1, w2, mu1, mu2, var1, var2)

    K = K - 1
    w_new = jnp.zeros((K,))
    

    mus_new = jnp.zeros((K,))
    

    vars_new = jnp.zeros((K,))
    

    # aux split trace = (j_star, r1, r2, u1, u2, u3)
    logQ_split = split_randomness_logQ(j_star, r1, r2, u1, u2, u3, K)

    X_new = {"w": w_new, "mus": mus_new, "vars": vars_new}

    log_alpha = model_log_prob(X_new) - lp + logQ_split - logQ_merge + detJ

    return jax.random.uniform(accept_key) < log_alpha
