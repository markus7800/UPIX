
from upix.all import *
import upix.distributions as dist
import matplotlib.pyplot as plt
import jax.numpy as jnp
from time import time
from upix.core.samplecontext import GenerateCtx
from upix.core.branching_tracer import retrace_branching

@model
def normal():
    x = sample("X", dist.Normal(0., 1.))
    factor(jax.lax.select(x < 0, 1., 2.))


m = normal()
slp = SLP_from_branchless_model(m)


def importance_sampling(slp: SLP):
    @jax.jit
    def _is(rng_key: PRNGKey):
        def _is_1(rng_key: PRNGKey):
            with GenerateCtx(rng_key) as ctx:
                slp.model()
                return ctx.X, ctx.log_likelihood
        _is_2 = retrace_branching(_is_1, slp.branching_decisions)
        (X, log_likelihood), path_condition = _is_2(rng_key)
        return X, jax.lax.select(path_condition, log_likelihood, -jnp.inf)
    return _is
        
is_func = importance_sampling(slp)
rng_keys = jax.random.split(jax.random.key(0), 10_000_000)
X, lp = jax.vmap(is_func)(rng_keys)
p = jnp.exp(lp - jax.scipy.special.logsumexp(lp))
plt.hist(X["X"], weights=p, bins=100, density=True)
xs = jnp.linspace(-5.,5., 1000)
ps = jnp.exp(dist.Normal(0.,1.).log_prob(xs) + jax.lax.select(xs < 0, jax.lax.full_like(xs,0.), jax.lax.full_like(xs,jnp.log(2.))))
ps = ps / jnp.trapezoid(ps, xs)
plt.plot(xs, ps)
plt.show()


n_chains = 10
n_samples_per_chain = 100_000

mcmc_config = MCMC(
    slp,
    MCMCStep(AllVariables(), HMC(10,0.1,unconstrained=False)),
    n_chains,
    collect_inference_info=True,
    progress_bar=True,
    return_map=lambda x: x.position
)

init_trace, init_log_prob = broadcast_jaxtree(m.generate(jax.random.key(0)), (mcmc_config.n_chains,))
init_trace = StackedTrace(init_trace, mcmc_config.n_chains)

result, all_positions = mcmc_config.run(jax.random.key(0), init_trace, init_log_prob, n_samples_per_chain=n_samples_per_chain)
assert result.infos is not None
for info in result.infos:
    print(summarise_mcmc_info(info, n_samples_per_chain))
print(result)

all_positions = StackedTraces(all_positions, n_samples_per_chain, mcmc_config.n_chains)

plt.hist(all_positions.unstack().data["X"], density=True, bins=100)
plt.plot(xs, ps)
plt.show()