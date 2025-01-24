from dccxjax import *
import jax
import numpyro.distributions as dist

@model
def simple():
    A = sample("A", dist.Normal(0.,1.0))
    if A > 0:
        B = sample("B", dist.Normal(0.3,1.3))
    else:
        C = sample("C", dist.Normal(0.7,2.7))


m: Model = simple() # type: ignore
print(m)

for key in range(10):
    print(f"{key=}")
    rng_key = jax.random.PRNGKey(key)
    slp = slp_from_prior(m, rng_key)
    print(slp)
    print("lp =", slp.log_prob(slp.decision_representative))
    print("lp =", slp.log_prob(slp.decision_representative))
    print(jax.make_jaxpr(slp._log_prob)(slp.decision_representative))