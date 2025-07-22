from dccxjax.all import *
import jax
import dccxjax.distributions as dist

@model
def simple():
    A = sample("A", dist.Normal(0.,1.0))
    if A > 0:
        B = sample("B", dist.Normal(0.3,1.3))
    else:
        C = sample("C", dist.Normal(0.7,2.7))

@model
def geometric(p):
    b = True
    i = 0
    while b:
        u = sample(f"u_{i}", dist.Uniform())
        b = u < p
        i += 1

@model
def selection():
    X = sample("X", dist.Normal(0.,1.0))
    Y = sample("Y", dist.Normal(0.,1.0))
    XY = [X,Y]
    A = sample("A", dist.Normal(0.,1.0))
    # i = jnp.astype(A > 0, int)
    i = (A > 0).astype(int)
    b = XY[i]
    # b = jax.lax.select(A > 0, X, Y)
    B = sample("B", dist.Normal(b,1.0))


@model
def poisson():
    p = sample("p", dist.Poisson(4))
    c = sample("c", dist.CategoricalProbs(jnp.full((p,), 1/p)))

# m: Model = simple() # type: ignore

m: Model = geometric(0.5) # type: ignore

# m: Model = selection() # type: ignore

# m: Model = poisson() # type: ignore

print(m)

logprob = make_model_logprob(m) # cannot jit this

active_slps: List[SLP] = []
for key in range(10):
    print(f"{key=}")
    rng_key = jax.random.PRNGKey(key)
    X = sample_from_prior(m, rng_key)
    print(X.keys())
    slp = slp_from_decision_representative(m, X)
    print(slp)
    print("lp =", slp.log_prob(slp.decision_representative))
    print("lp =", slp.log_prob(slp.decision_representative))

    # print(jax.make_jaxpr(slp._log_prob)(slp.decision_representative))

    if all(slp.path_indicator(X) == 0 for slp in active_slps):
        print("new path!")
        active_slps.append(slp)

print()
print()
print("active_slps: ")
for slp in active_slps:
    print(slp)
    # print(jax.make_jaxpr(slp.log_prob)(slp.decision_representative))