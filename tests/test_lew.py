# lew 2023 Prop 5.1
#%%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from typing import Callable

def make_mh_kernel(k:Callable[[jax.Array],dist.Distribution], log_P:Callable[[jax.Array], jax.Array]):
    def mh_kernel(x:jax.Array, rng_key: jax.Array,) -> jax.Array:
        propose_key, accept_key = jax.random.split(rng_key)
        forward_dist = k(x)
        proposed_x = forward_dist.sample(propose_key)
        backward_dist = k(proposed_x)
        A = log_P(proposed_x) - log_P(x) + backward_dist.log_prob(x) - forward_dist.log_prob(proposed_x)
        u = jax.random.uniform(accept_key)
        return jax.lax.select(jax.lax.log(u) < A, proposed_x, x)
    return mh_kernel
    
def mixture_log_P(x:jax.Array):
    return jax.lax.log(
        0.3 * jax.lax.exp(dist.Normal(-1.,0.5).log_prob(x)) +
        0.7 * jax.lax.exp(dist.Normal(2., 0.5).log_prob(x))
    )

def mixture_sample(rng_key: jax.Array, N: int):
    key1, key2, key3 = jax.random.split(rng_key,3)
    z1 = dist.Normal(-1.,0.5).sample(key1, (N,))
    z2 = dist.Normal(2., 0.5).sample(key2, (N,))
    b = dist.BernoulliProbs(0.3).sample(key3, (N,))
    return jax.lax.select(b,z1,z2)

# %%
xs = jnp.linspace(-4,4,500)
ps = jax.lax.exp(mixture_log_P(xs))
plt.plot(xs, ps)
plt.show()
# %%
x_sample = mixture_sample(jax.random.key(0), 1_000_000)
plt.hist(x_sample, density=True, bins=100, alpha=0.5, color="tab:blue")
plt.plot(xs, ps, color="tab:blue")
plt.show()

# %%
# mh kernel satisfies detailed balance (it is its own time reversal), leaves p in variant
# general reversal k(x,x') = k(x',x) p(x') / p(x) (annealed importance sampling paper)
mh_kernel = jax.vmap(make_mh_kernel(lambda x: dist.Normal(x, 1.), mixture_log_P))

#%%
x_mh_sample = mh_kernel(x_sample, jax.random.split(jax.random.key(0), x_sample.size))
plt.hist(x_mh_sample, density=True, bins=100, alpha=0.5, color="tab:blue")
plt.plot(xs, ps, color="tab:blue")
plt.show()
# %%
def Q_sample(rng_key: jax.Array, N: int):
    return dist.Normal(0.,0.1).sample(rng_key, (N,))

def Q_log(x: jax.Array):
    return dist.Normal(0.,0.1).log_prob(x)

x_sample = mixture_sample(jax.random.key(0), 1_000_000)
plt.hist(x_sample, density=True, bins=100, alpha=0.5, color="tab:blue")
plt.plot(xs, ps, color="tab:blue")
q_sample = Q_sample(jax.random.key(0), 1_000_000)
plt.hist(q_sample, density=True, bins=100, alpha=0.5, color="tab:green")
qs = jax.lax.exp(Q_log(xs))
plt.plot(xs, qs, color="tab:green")
plt.show()
# %%
def apply_mh_kernel_n(sample: jax.Array, rng_key: jax.Array, n: int, mh_kernel) -> jax.Array:
    # if n == 0:
    #     return sample
    
    # rng_key, n_key = jax.random.split(rng_key)
    # sample = apply_mh_kernel_n(sample, n_key, n-1, mh_kernel)

    # return mh_kernel(sample, jax.random.split(rng_key, x_sample.size))

    return jax.lax.scan(
        lambda xs, rng_key: (mh_kernel(xs, jax.random.split(rng_key, xs.size)), None),
        sample,
        jax.random.split(rng_key, n)
        )[0]

N = 100
q_mh_sample = apply_mh_kernel_n(q_sample, jax.random.key(0), N, mh_kernel)
plt.hist(q_mh_sample, density=True, bins=100)
plt.plot(xs, ps)
plt.plot(xs, qs, color="tab:green")
plt.show()
# %%
def est_q_mh_log_prob(xs: jax.Array, n: int, mh_kernel, rng_key: jax.Array, m: int,
                      log_Q: Callable[[jax.Array], jax.Array], log_P: Callable[[jax.Array], jax.Array]):
    
    def _est_q_mh_log_prob(ps: jax.Array, rng_key: jax.Array):
        xs0 = apply_mh_kernel_n(xs, rng_key, n, mh_kernel)
        return ps + jax.lax.exp(log_Q(xs0) + log_P(xs) - log_P(xs0)), None

    ps, _ = jax.lax.scan(_est_q_mh_log_prob, jax.lax.zeros_like_array(xs), jax.random.split(rng_key,m))
    return ps/ m

qs_mh = est_q_mh_log_prob(xs, N, mh_kernel, jax.random.key(0), 1_000, Q_log, mixture_log_P)

plt.hist(q_mh_sample, density=True, bins=100, alpha=0.5, color="tab:blue")
plt.plot(xs, qs_mh, color="tab:blue")
plt.show()
# %%
qs_mh = est_q_mh_log_prob(xs, 1, mh_kernel, jax.random.key(0), 1_000, mixture_log_P, mixture_log_P)

plt.hist(x_sample, density=True, bins=100, alpha=0.5, color="tab:blue")
plt.plot(xs, qs_mh, color="tab:blue")
plt.show()
# %%
# estimate z
T = 1_000
posterior_approx = mixture_sample(jax.random.key(0), T)

def IS_Q_sample(posterior_approx: jax.Array, rng_key: jax.Array) -> jax.Array:
    return dist.Normal(posterior_approx, 1.).sample(rng_key)

def get_IS_log_Q(posterior_approx: jax.Array, mix: bool):
    def log_Q(x: jax.Array) -> jax.Array:
        if mix:
            log_qs = dist.Normal(posterior_approx.reshape(-1,1), 1.).log_prob(x.reshape(1,-1))
            return jax.scipy.special.logsumexp(log_qs, axis=0) - jnp.log(posterior_approx.size)
        else:
            return dist.Normal(posterior_approx, 1.).log_prob(x)
    return log_Q
# %%
is_q_sample = IS_Q_sample(posterior_approx, jax.random.key(0))
log_qs = get_IS_log_Q(posterior_approx, True)(is_q_sample)
jax.lax.exp(jax.scipy.special.logsumexp(mixture_log_P(is_q_sample) - log_qs)) / is_q_sample.size
# %%
M = 100
is_q_sample = jax.vmap(IS_Q_sample, (None,0))(posterior_approx, jax.random.split(jax.random.key(0),M))
log_qs = jax.vmap(get_IS_log_Q(posterior_approx, True))(is_q_sample)
jax.lax.exp(jax.scipy.special.logsumexp(mixture_log_P(is_q_sample) - log_qs)) / is_q_sample.size
# %%
is_q_mh_sample = mh_kernel(IS_Q_sample(posterior_approx, jax.random.key(0)), jax.random.split(jax.random.key(0),T))
is_q_mh_sample_0 = mh_kernel(is_q_mh_sample, jax.random.split(jax.random.key(0),T))
log_qs = mixture_log_P(is_q_mh_sample) - (get_IS_log_Q(posterior_approx, True)(is_q_mh_sample_0) + mixture_log_P(is_q_mh_sample) - mixture_log_P(is_q_mh_sample_0))
log_qs = mixture_log_P(is_q_mh_sample_0) - get_IS_log_Q(posterior_approx, True)(is_q_mh_sample_0)
jax.lax.exp(jax.scipy.special.logsumexp(log_qs)) / is_q_mh_sample.size
# %%
