from dccxjax import *
import jax
import numpyro.distributions as dist

@model
def simple():
    A = sample("A", dist.Normal(0.,1.))
    if A > 0:
        B = sample("B", dist.Normal(0.,1.))
    else:
        C = sample("C", dist.Normal(0.,1.))


m: Model = simple() # type: ignore
print(m)

for key in range(10):
    print(f"{key=}")
    rng_key = jax.random.PRNGKey(key)
    slp = slp_from_prior(m, rng_key)
    print(slp)
    print("lp =", slp.log_prob(slp.decision_representative))