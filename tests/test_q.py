import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt

# def Q(rng_key: jax.Array):
#     y = dist.Normal(0,1).sample(rng_key)
#     return y, dist.Normal(0,1).log_prob(y)

def Q(rng_key: jax.Array):
    key1, key2 = jax.random.split(rng_key)
    x = dist.Normal(0,1).sample(key1)
    y = dist.Normal(x,1).sample(key2)
    return y, dist.Normal(0,1).log_prob(x) + dist.Normal(x,1).log_prob(y)


ys, qs = jax.vmap(Q)(jax.random.split(jax.random.key(0), 100_000_000))

plt.hist(ys, bins=100, density=True)
xs = jnp.linspace(-5,5,1000)
plt.plot(xs, jnp.exp(dist.Normal(0, jnp.sqrt(2)).log_prob(xs)))
plt.show()

P = dist.Normal(1,0.5)

qs = dist.Normal(0, jnp.sqrt(2)).log_prob(ys) # correct marginal prob for ys

plt.plot(xs, jnp.exp(P.log_prob(xs)))
log_weights = P.log_prob(ys) - qs
W = jax.scipy.special.logsumexp(log_weights)
weights = jnp.exp(log_weights - W)
print(jnp.sum(ys * weights), jnp.sum(weights))

plt.hist(ys, weights = weights, bins=100, density=True)
plt.show()