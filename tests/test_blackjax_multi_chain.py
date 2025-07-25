import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from time import time

loc, scale = 10, 20
observed = loc + scale * jax.random.normal(jax.random.key(0), (1000,))


def logdensity_fn(loc, log_scale, observed=observed):
    """Univariate Normal"""
    scale = jnp.exp(log_scale)
    logjac = log_scale
    logpdf = stats.norm.logpdf(observed, loc, scale)
    return logjac + jnp.sum(logpdf)


def logdensity(x):
    return logdensity_fn(**x)


def inference_loop(rng_key, kernel, initial_state, num_samples):

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


import blackjax
num_chains = 10

inv_mass_matrix = jnp.array([0.5, 0.01])
step_size = 1e-3
num_integration_steps = 10

hmc = blackjax.hmc(logdensity, step_size, inv_mass_matrix, num_integration_steps)

def inference_loop_multiple_chains(
    rng_key, kernel, initial_state, num_samples, num_chains
):

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, _ = jax.vmap(kernel)(keys, states)
        return states, states

    keys = jax.random.split(rng_key, num_samples)
    state, states = jax.lax.scan(one_step, initial_state, keys)

    return state

initial_positions = {"loc": jnp.ones(num_chains), "log_scale": jnp.ones(num_chains)}
initial_states = jax.vmap(hmc.init, in_axes=(0))(initial_positions)

t0 = time()
state = inference_loop_multiple_chains(
    jax.random.key(0), hmc.step, initial_states, 100_000, num_chains
)
_ = state.position["loc"].block_until_ready()
t1 = time()
print(f"Finished in {t1-t0:.3f}s")
print(state.position["loc"])