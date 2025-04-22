import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from dccxjax import PRNGKey
from tqdm import tqdm
from data import *

def get_lw_log_weight(K):
    @jax.jit
    def lw(rng_key: PRNGKey):
        w_key, mus_key, vars_key = jax.random.split(rng_key,3)

        w = dist.Dirichlet(jnp.full((K+1,), delta)).sample(w_key)
        mus = dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))).sample(mus_key)
        vars = dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).sample(vars_key)

        log_likelihoods = jnp.log(jnp.sum(w.reshape(1,-1) * jnp.exp(dist.Normal(mus.reshape(1,-1), jnp.sqrt(vars).reshape(1,-1)).log_prob(ys.reshape(-1,1))), axis=1))

        return log_likelihoods.sum()
    return lw

def get_lw_log_weight_2(K):
    @jax.jit
    def lw(rng_key: PRNGKey):
        w_key, mus_key, vars_key, zs_key = jax.random.split(rng_key,4)

        w = dist.Dirichlet(jnp.full((K+1,), delta)).sample(w_key)
        mus = dist.Normal(jnp.full((K+1,), xi), jnp.full((K+1,), 1/jax.lax.sqrt(kappa))).sample(mus_key)
        vars = dist.InverseGamma(jnp.full((K+1,), alpha), jnp.full((K+1,), beta)).sample(vars_key)

        zs = dist.CategoricalProbs(w).sample(zs_key, ys.shape)

        log_likelihoods = dist.Normal(mus[zs], jnp.sqrt(vars[zs])).log_prob(ys)

        return log_likelihoods.sum()
    return lw

def IS(log_weight, K: int, N: int, batch_method: int = 1):

    batch_size = 1_000_000
    if N > batch_size and batch_method > 0:
        if batch_method == 1:
            assert N % batch_size == 0
            N_batches = N // batch_size

            def batch(rng_key: PRNGKey):
                keys = jax.random.split(rng_key, batch_size)
                log_Ws = jax.vmap(log_weight)(keys)
                return (jax.scipy.special.logsumexp(log_Ws), jax.scipy.special.logsumexp(log_Ws * 2))

            batch_keys = jax.random.split(jax.random.PRNGKey(0), N_batches)
            log_W, log_W_squared = jax.lax.map(batch, batch_keys)
            log_Z = jax.scipy.special.logsumexp(log_W) - jnp.log(N_batches) - jnp.log(batch_size)
            ESS = jnp.exp(jax.scipy.special.logsumexp(log_W)*2 - jax.scipy.special.logsumexp(log_W_squared))
        else:
            keys = jax.random.split(jax.random.PRNGKey(0), N)
            log_W = jax.lax.map(log_weight, keys, batch_size=batch_size)
            log_Z = jax.scipy.special.logsumexp(log_W) - jnp.log(N_batches) - jnp.log(batch_size)
            ESS = jnp.exp(jax.scipy.special.logsumexp(log_W)*2 - jax.scipy.special.logsumexp(log_W*2))
    else:
        keys = jax.random.split(jax.random.PRNGKey(0), N)
        log_W = jax.vmap(log_weight)(keys)
        log_Z = jax.scipy.special.logsumexp(log_W) - jnp.log(N)
        ESS = jnp.exp(jax.scipy.special.logsumexp(log_W)*2 - jax.scipy.special.logsumexp(log_W*2))
        
    return log_Z, ESS

def do_lw(N = 1_000_000):
    batch_method=1
    print(f"do_lw {N=:,} {batch_method=} {ys.shape=}")
    Ks = range(0, 7)
    result = [IS(get_lw_log_weight(K), K, N, batch_method=batch_method) for K in tqdm(Ks)]
    log_Z_path = jnp.array([r[0] for r in result])
    ESS = jnp.array([r[1] for r in result])
    print(f"{log_Z_path=}")
    print(f"{ESS=}")
    # print(f"{jnp.exp(log_Z_path - jax.scipy.special.logsumexp(log_Z_path))}")

    log_Z_path_prior = dist.Poisson(lam).log_prob(jnp.array(list(Ks)))
    log_Z = log_Z_path + log_Z_path_prior
    path_weight = jnp.exp(log_Z - jax.scipy.special.logsumexp(log_Z))

    print(f"do_lw {N=:,} {batch_method=} {ys.shape=}")
    for i, k in enumerate(Ks):
        print(k, path_weight[i])