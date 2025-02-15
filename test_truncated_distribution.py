import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt


d = dist.TwoSidedTruncatedDistribution(dist.Normal(1.,1.), 0., 3., validate_args=True)

xs = jnp.linspace(-5.,5.,500)
plt.plot(xs, jnp.exp(d.log_prob(xs)))
plt.show()


d = dist.TwoSidedTruncatedDistribution(dist.Normal(1.,1.), 0., jnp.inf, validate_args=True)

xs = jnp.linspace(-5.,5.,500)
plt.plot(xs, jnp.exp(d.log_prob(xs)))
plt.show()