#%%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from typing import Callable
from time import time

def make_mh_kernel(k:Callable[[jax.Array],dist.Distribution], log_p:Callable[[jax.Array], jax.Array]):
    @jax.jit
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

def sample_prior(rng_key: jax.Array, N: int) -> jax.Array:
    return dist.Normal(0., 1.).sample(rng_key, (N,)) # type: ignore
def log_prior(xs: jax.Array) -> jax.Array:
    return dist.Normal(0., 1.).log_prob(xs)


# okay prior
# obs_sigma = 1.
# obs = 2.
# xlims = (-5.,5.)

# miss-specified prior
obs_sigma = 0.1
obs = 4.
xlims = (3.,5.)

def log_joint(xs: jax.Array) -> jax.Array:
    return log_prior(xs) + dist.Normal(xs, obs_sigma).log_prob(obs)

def get_posterior():
    sigma = jnp.sqrt(1 / (1 + 1/obs_sigma**2))
    mu = sigma**2 * obs / obs_sigma**2
    return dist.Normal(mu, sigma)

def log_posterior(xs: jax.Array) -> jax.Array:
    return get_posterior().log_prob(xs)

def sample_posterior(rng_key: jax.Array, N: int) -> jax.Array:
    return get_posterior().sample(rng_key, (N,)) # type:ignore

x_range = jnp.linspace(*xlims, 1000)
bins = jnp.linspace(*xlims, 100)


ps = jnp.exp(log_joint(x_range))
true_Z = jnp.trapezoid(ps, x_range).item()
print(f"{true_Z=}")
ps = ps / true_Z
# plt.plot(x_range, ps)
# plt.plot(x_range, jnp.exp(log_posterior(x_range)), linestyle="-.")
# plt.title("True posterior")
# plt.show()

N = 1_000_000

from dccxjax.all import *
import dccxjax.distributions as dist
import dccxjax.distributions.constraints as constraints
from dccxjax.infer.optimizers import Adagrad

@model
def normal():
    x = sample("x", dist.Normal(0, 1))
    sample("y", dist.Normal(x, obs_sigma), observed=obs)

m: Model = normal()
slp = SLP_from_branchless_model(m)


@guide
def normal_guide():
    mu = param("mu", jnp.array(0.,float))
    sigma = param("sigma", jnp.array(1.,float), constraints.positive)
    sample("x", dist.Normal(mu, sigma))
    

g: Guide = normal_guide()

g: Guide = MeanfieldNormalGuide(slp, AllVariables())

print(g.sample_and_log_prob(jax.random.PRNGKey(0), ()))
x, lp = g.sample_and_log_prob(jax.random.PRNGKey(0), (10,3))
print(x["x"].shape, lp.shape)

advi = ADVI(slp, AllVariables(), g, Adagrad(1.), 100, progress_bar=True)

last_state, elbo = advi.run(jax.random.PRNGKey(0), n_iter=1_000)
plt.figure()
plt.plot(elbo)
# plt.show()

g = advi.get_updated_guide(last_state)
posterior = g.sample(jax.random.PRNGKey(0), (1_000_000,))
plt.figure()
plt.hist(posterior["x"], bins=100, density=True)
plt.plot(x_range, ps)
plt.plot(x_range, jax.lax.exp(jax.vmap(g.log_prob)({"x": x_range})))
plt.show()
