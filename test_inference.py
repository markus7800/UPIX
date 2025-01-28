from dccxjax import *
import numpyro.distributions as dist

@model
def normal():
    X = sample("X", dist.Normal(0.,1.))
    Y = sample("Y", dist.Normal(X, 1.))


m: Model = normal()
slp = convert_model_to_SLP(m)

result = mcmc(
    slp,
    InferenceStep(AllVariables(), RW(gaussian_random_walk(0.5))),
    10,
    1,
    jax.random.PRNGKey(0)
    ),

print(result)