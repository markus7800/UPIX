# lew 2023 Prop 5.1
#%%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from typing import Callable

def make_mh_kernel(k:Callable[[jax.Array],dist.Distribution], log_p:Callable[[jax.Array], jax.Array]):
    def mh_kernel(x:jax.Array, rng_key: jax.Array,) -> jax.Array:
        propose_key, accept_key = jax.random.split(rng_key)
        forward_dist = k(x)
        proposed_x = forward_dist.sample(propose_key)
        backward_dist = k(proposed_x)
        A = log_p(proposed_x) - log_p(x) + backward_dist.log_prob(x) - forward_dist.log_prob(proposed_x)
        u = jax.random.uniform(accept_key)
        return jax.lax.select(jax.lax.log(u) < A, proposed_x, x)
    return mh_kernel

def apply_mh_kernel_n(sample: jax.Array, rng_key: jax.Array, n: int, mh_kernel) -> jax.Array:
    return jax.lax.scan(
        lambda xs, rng_key: (mh_kernel(xs, jax.random.split(rng_key, xs.size)), None),
        sample,
        jax.random.split(rng_key, n)
        )[0]

def sample_prior(rng_key: jax.Array, N: int):
    return dist.Normal(0., 1.).sample(rng_key, (N,))
def log_prior(xs: jax.Array):
    return dist.Normal(0., 1.).log_prob(xs)


# def okay_log_P(xs: jax.Array):
#     return log_prior(xs) + dist.Normal(xs, 1.).log_prob(2.)
# xlims = (-5.,5.)
# x_range = jnp.linspace(-5.,5.,1000)
# bins = jnp.linspace(-5.,5.,100)
# log_P = okay_log_P

def misspecified_log_P(xs: jax.Array):
    return log_prior(xs) + dist.Normal(xs, 0.1).log_prob(4.)
xlims = (3.,5.)
x_range = jnp.linspace(3.,5.,1000)
bins = jnp.linspace(3.,5.,100)
log_P = misspecified_log_P


ps = jnp.exp(log_P(x_range))
true_Z = jnp.trapezoid(ps, x_range).item()
print(f"{true_Z=}")
ps = ps / true_Z
# plt.plot(x_range, ps)
# plt.title("True posterior")
# plt.show()

N = 1_000_000


def get_Z_ESS(log_weights):
    Z = jnp.exp(jax.scipy.special.logsumexp(log_weights)) / N
    ESS = jnp.exp(jax.scipy.special.logsumexp(log_weights)*2 - jax.scipy.special.logsumexp(log_weights*2))
    return Z, ESS


# likelihood weighting
def likelihood_weighting():
    x_sample = sample_prior(jax.random.PRNGKey(0), N)
    log_weights = log_P(x_sample) - log_prior(x_sample)
    Z, ESS = get_Z_ESS(log_weights)
    print(f"LW: {log_weights=} {x0=} {Z=} {ESS=}")

    plt.hist(x_sample, weights=jnp.exp(log_weights), density=True, bins=bins)
    plt.xlim(xlims)
    plt.plot(x_range, ps)
    plt.title(f"Likelihood weighting Z={Z.item()} ESS={ESS.item()}")
    plt.show()

# likelihood_weighting()

# zero step AIS
# beta_0 = 1   f0 un-normalised posterior
# beta_1 = 0   f1 = f0^0 * f1^(1-0)  f1 = fn = prior
# = likelihood weighting

# one step AIS
# beta_0 = 1    f0 un-normalised posterior
# beta_1 = 0.5  f1 = f0^0.5 * f2*0.5
# beta_2 = 0    f2 = f0^0 * f2^(1-0)  f2 = fn = prior
# T_j leaves p_j invariant p_j = f_j / c_j normalised

def one_step_AIS():
    log_f2 = log_prior
    x1 = sample_prior(jax.random.PRNGKey(0), N)
    log_f1 = lambda x: 0.5*log_P(x) + 0.5*log_prior(x)
    T1 = jax.vmap(make_mh_kernel(lambda x: dist.Normal(x, 1.), log_f1))
    # x0 = T1(x1, jax.random.split(jax.random.PRNGKey(0), N))
    rng_key, mh_key = jax.random.split(jax.random.PRNGKey(0))
    x0 = apply_mh_kernel_n(x1, mh_key, 1, T1)
    log_f0 = log_P
    xs = x0

    log_weights = (log_f1(x1) - log_f2(x1)) + (log_f0(x0) - log_f1(x0))
    Z, ESS = get_Z_ESS(log_weights)

    # print(f"One step AIS: {log_weights=} {x0=} {Z=} {ESS=}")

    plt.hist(xs, weights=jnp.exp(log_weights), density=True, bins=bins)
    plt.plot(x_range, ps)
    plt.xlim(xlims)
    plt.title(f"One step AIS beta=0.5 Z={Z.item()} ESS={ESS.item()}")
    plt.show()

# one_step_AIS()

# one BIG-step AIS

def one_big_step_AIS():
    log_f2 = log_prior
    x1 = sample_prior(jax.random.PRNGKey(0), N)

    log_f1 = lambda x: 0.5*log_P(x) + 0.5*log_prior(x)
    T1 = jax.vmap(make_mh_kernel(lambda x: dist.Normal(x, 1.), log_f1))
    x0 = apply_mh_kernel_n(x1, jax.random.PRNGKey(0), 100, T1)
    log_f0 = log_P

    plt.hist(x0, density=True, bins=bins)
    ps1 = jnp.exp(log_f1(x_range))
    ps1 = ps1 / (ps1.sum() * (x_range[1] - x_range[0]))
    plt.plot(x_range, ps1)
    plt.xlim(xlims)
    plt.title("x0 approximates p1=f1/c1")
    plt.show()

    xs = x0
    log_weights = (log_f1(x1) - log_f2(x1)) + (log_f0(x0) - log_f1(x0))
    Z, ESS = get_Z_ESS(log_weights)

    plt.hist(xs, weights=jnp.exp(log_weights), density=True, bins=bins)
    plt.plot(x_range, ps)
    plt.xlim(xlims)
    plt.title(f"One big step AIS beta=0.5 Z={Z.item()} ESS={ESS.item()}")
    plt.show()

# one_big_step_AIS()


# one BIG-step to almost posterior AIS
# beta_0 = 1    f0 un-normalised posterior
# beta_1 = 0.99  f1 = f0^0.99 * f2*0.01
# beta_2 = 0    f2 = f0^0 * f2^(1-0)  f2 = fn = prior

def one_big_step_to_almost_posterior_AIS():
    log_f2 = log_prior
    x1 = sample_prior(jax.random.PRNGKey(0), N)

    log_f1 = lambda x: 0.99*log_P(x) + 0.01*log_prior(x)
    T1 = jax.vmap(make_mh_kernel(lambda x: dist.Normal(x, 1.), log_f1))
    x0 = apply_mh_kernel_n(x1, jax.random.PRNGKey(0), 100, T1) # x0 is MC estimate of almost posterior
    log_f0 = log_P

    plt.hist(x0, density=True, bins=bins)
    ps1 = jnp.exp(log_f1(x_range))
    ps1 = ps1 / (ps1.sum() * (x_range[1] - x_range[0]))
    plt.plot(x_range, ps1)
    plt.plot(x_range, ps)
    plt.xlim(xlims)
    plt.title("x0 approximates almost posterior ")
    plt.show()

    xs = x0 
    log_weights = (log_f1(x1) - log_f2(x1)) + (log_f0(x0) - log_f1(x0))
    Z, ESS = get_Z_ESS(log_weights)

    print("log_f0(x0) - log_f1(x0) should be close to 0:", jnp.max(jnp.abs(log_f0(x0) - log_f1(x0))))
    print("log_f1(x1) - log_f2(x1)) should be close to likelihood of sample from prior:", jnp.max(jnp.abs(log_P(x1) - log_prior(x1) - (log_f1(x1) - log_f2(x1)))))


    plt.hist(xs, weights=jnp.exp(log_weights), density=True, bins=bins)
    plt.plot(x_range, ps)
    plt.xlim(xlims)
    plt.title(f"One big step AIS beta=0.99 Z={Z.item()} ESS={ESS.item()}")
    plt.show()

# one_big_step_to_almost_posterior_AIS()

# https://random-walks.org/book/papers/ais/ais.html


def AIS(betas, M, plot=False):
    from typing import NamedTuple
    class AISCarry(NamedTuple):
        xs: jax.Array
        log_weights: jax.Array
        rng_key: jax.Array

    def step(carry: AISCarry, beta: float):
        log_f = lambda x: beta*log_P(x) + (1-beta)*log_prior(x)
        T = jax.vmap(make_mh_kernel(lambda x: dist.Normal(x, 1.), log_f))
        rng_key, mh_key = jax.random.split(carry.rng_key)
        xs = apply_mh_kernel_n(carry.xs, mh_key, M, T)
        log_weights = carry.log_weights + log_f(carry.xs) - log_f(xs)
        return AISCarry(xs, log_weights, rng_key), None

    xn = sample_prior(jax.random.PRNGKey(0), N)
    log_weights = -log_prior(xn)
    last_state, _ = jax.lax.scan(step, AISCarry(xn, log_weights, jax.random.PRNGKey(0)), betas)
    x0 = last_state.xs
    log_weights = last_state.log_weights + log_P(x0)
    Z, ESS = get_Z_ESS(log_weights)


    print(f"Scan: Z={Z.item()} ESS={ESS.item()}")

    if plot:
        plt.hist(x0, weights=jnp.exp(log_weights), density=True, bins=bins)
        plt.plot(x_range, ps)
        plt.xlim(xlims)
        plt.title(f"Scan AIS Z={Z.item()} ESS={ESS.item()}")
        plt.show()



# AIS(jnp.array([0.5]), 1)


def sigmoid(z):
    return 1/(1 + jnp.exp(-z))

# plt.plot())
# plt.show()

AIS(jnp.linspace(0.1,0.99, 100), 1)

a = 5
AIS(sigmoid(jnp.linspace(-a,a,100)), 1, True)

a = 10
AIS(sigmoid(jnp.linspace(-a,a,100)), 1, True)

a = 25
AIS(sigmoid(jnp.linspace(-a,a,1000)), 1, True)

a = 25
AIS(sigmoid(jnp.linspace(-a,a,1000)), 10, True)
