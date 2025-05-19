from dccxjax.core import SLP
import jax
from ..types import PRNGKey, Trace
from typing import NamedTuple
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

__all__ = [
    "coordinate_ascent",
    "simulated_annealing",
    "sparse_coordinate_ascent",
    "sparse_coordinate_ascent2",
]

class CoordianteAscentState(NamedTuple):
    i: jax.Array
    position: Trace
    log_prob: jax.Array

def coordinate_ascent(slp: SLP, scale: float, n_iter: int, n_chains: int, seed: PRNGKey):
    @jax.jit
    def step(state: CoordianteAscentState, rng_key: PRNGKey):
        position = state.position
        log_prob = state.log_prob
        for addr, value in position.items():
            rng_key, sample_key = jax.random.split(rng_key)
            position[addr] = value + scale * jax.random.normal(sample_key, value.shape) # 1/jax.lax.sqrt(state.i)
            new_log_prob = slp._log_prob(position)
            position[addr] = jax.lax.select(new_log_prob > log_prob, position[addr], value)
            log_prob = jax.lax.max(new_log_prob, log_prob)

        return CoordianteAscentState(state.i + 1, position, log_prob), position
    
    keys = jax.random.split(seed, (n_iter, n_chains))

    X = slp.decision_representative
    lp = slp._log_prob(slp.decision_representative)
    first_state = CoordianteAscentState(jax.lax.broadcast(1., (n_chains,)), jax.tree.map(lambda x: jax.lax.broadcast(x, (n_chains,)), X), jax.lax.broadcast(lp, (n_chains,)))
    last_state, result = jax.lax.scan(jax.vmap(step), first_state, keys)

    return last_state.position, last_state.log_prob, result


class SimulatedAnnealingState(NamedTuple):
    iteration: jax.Array
    temp: jax.Array
    flat_position: jax.Array
    log_prob: jax.Array
    best_flat_position: jax.Array
    best_log_prob: jax.Array

def simulated_annealing(slp: SLP, temp: float, n_iter: int, n_chains: int, seed: PRNGKey):
    n_variables = len(slp.decision_representative)
    var_perm = jnp.arange(0, n_variables)

    flat_X, unravel_fn = ravel_pytree(slp.decision_representative)
    lp = slp._log_prob(slp.decision_representative)

    def _coordinate_update(i: int, a: SimulatedAnnealingState, body_key: PRNGKey):
        body_key, sample_key, accept_key = jax.random.split(body_key, 3)
        old_value = a.flat_position[i]
        new_value = old_value + a.temp * jax.random.normal(sample_key)
        new_flat_position = a.flat_position.at[i].set(new_value)
        new_log_prob = slp._log_prob(unravel_fn(new_flat_position))

        log_alpha = jax.lax.log(jax.random.uniform(accept_key))
        accept = log_alpha < (new_log_prob - a.log_prob)
        next_flat_position = jax.lax.select(accept, new_flat_position, a.flat_position)
        next_log_prob = jax.lax.select(accept, new_log_prob, a.log_prob)

        new_best = new_log_prob > a.best_log_prob
        best_flat_position = jax.lax.select(new_best, new_flat_position, a.best_flat_position)
        best_log_prob = jax.lax.select(new_best, new_log_prob, a.best_log_prob)
        
        return SimulatedAnnealingState(a.iteration + 1, a.temp, next_flat_position, next_log_prob, best_flat_position, best_log_prob), body_key

    @jax.jit
    def step(state: SimulatedAnnealingState, rng_key: PRNGKey):
        next_state, _ = jax.lax.fori_loop(0, n_variables, lambda i, a: _coordinate_update(i, *a), (state, rng_key))
        return next_state, unravel_fn(next_state.flat_position)
    

    first_state = SimulatedAnnealingState(1, temp, flat_X, lp, flat_X, lp) # type: ignore
    first_states = jax.tree.map(lambda x: jax.lax.broadcast(x, (n_chains,)), first_state)
    
    keys = jax.random.split(seed, (n_iter, n_chains))
    last_state, result = jax.lax.scan(jax.vmap(step), first_states, keys)

    return jax.vmap(unravel_fn)(last_state.best_flat_position), last_state.best_log_prob, result



class SparseCoordianteAscentState(NamedTuple):
    flat_position: jax.Array
    log_prob: jax.Array

def sparse_coordinate_ascent(slp: SLP, scale: float, p: float, n_iter: int, n_chains: int, seed: PRNGKey):

    flat_X, unravel_fn = ravel_pytree(slp.decision_representative)
    lp = slp._log_prob(slp.decision_representative)

    @jax.jit
    def step(state: SparseCoordianteAscentState, rng_key: PRNGKey):
        rng_key, sample_key, mask_key = jax.random.split(rng_key, 3)
        Z = jax.random.normal(sample_key, state.flat_position.shape)
        mask = jax.random.bernoulli(mask_key, p, state.flat_position.shape)
        new_flat_position = state.flat_position + scale * mask * Z
        new_log_prob = slp._log_prob(unravel_fn(new_flat_position))

        next_flat_position = jax.lax.select(new_log_prob > state.log_prob, new_flat_position, state.flat_position)
        next_log_prob = jax.lax.max(new_log_prob, state.log_prob)

        return SparseCoordianteAscentState(next_flat_position, next_log_prob), unravel_fn(next_flat_position)
    

    first_state = SparseCoordianteAscentState(flat_X, lp) # type: ignore
    first_states = jax.tree.map(lambda x: jax.lax.broadcast(x, (n_chains,)), first_state)
    
    keys = jax.random.split(seed, (n_iter * int(jax.lax.ceil(1. / p)), n_chains))
    last_state, result = jax.lax.scan(jax.vmap(step), first_states, keys)

    return jax.vmap(unravel_fn)(last_state.flat_position), last_state.log_prob, result



class SparseCoordianteAscentState2(NamedTuple):
    position: Trace
    log_prob: jax.Array

def sparse_coordinate_ascent2(slp: SLP, scale: float, p: float, n_iter: int, n_chains: int, seed: PRNGKey):

    X = slp.decision_representative
    lp = slp._log_prob(slp.decision_representative)

    all_continuous = slp.all_continuous()
    # 
    # is_discrete_X = {addr: jax.lax.full_like(val, is_discrete[addr]) for addr, val in slp.decision_representative.items()}
    # assert list(slp.decision_representative.keys()) == list(is_discrete_X.keys())
    # flat_is_discrete, _ = ravel_pytree(is_discrete_X) # is hopefully the same structure as flat_X
    # print(f"{flat_is_discrete=}")
    @jax.jit
    def step(state: SparseCoordianteAscentState2, rng_key: PRNGKey):
        rng_key, sample_key, mask_key = jax.random.split(rng_key, 3)
        if all_continuous:
            flat_position, unravel_fn = ravel_pytree(state.position)
            Z = jax.random.normal(sample_key, flat_position.shape)
            mask = jax.random.bernoulli(mask_key, p, flat_position.shape)
            new_flat_position = flat_position + scale * mask * Z

            new_position = unravel_fn(new_flat_position)

        else:
            position = slp.transform_to_unconstrained(state.position)

            is_discrete = slp.get_is_discrete_map()
            discrete_position = {addr: val for addr, val in position.items() if is_discrete[addr]}
            flat_discrete_position, discrete_unravel_fn = ravel_pytree(discrete_position)
            B = 2 * jax.random.bernoulli(sample_key, 0.5, flat_discrete_position.shape) - 1
            mask = jax.random.bernoulli(mask_key, p, flat_discrete_position.shape)
            new_flat_discrete_position = flat_discrete_position * mask * B

            continuous_position = {addr: val for addr, val in position.items() if not is_discrete[addr]}
            flat_continuous_position, continuous_unravel_fn = ravel_pytree(continuous_position)
            Z = jax.random.normal(sample_key, flat_continuous_position.shape)
            mask = jax.random.bernoulli(mask_key, p, flat_continuous_position.shape)
            new_flat_continuous_position = flat_continuous_position + scale * mask * Z

            new_position = slp.transform_to_constrained(discrete_unravel_fn(new_flat_discrete_position) | continuous_unravel_fn(new_flat_continuous_position))
        
        new_log_prob = slp._log_prob(new_position)

        next_position = jax.lax.cond(new_log_prob > state.log_prob, lambda _: new_position, lambda _: state.position, operand=None)
        next_log_prob = jax.lax.max(new_log_prob, state.log_prob)

        return SparseCoordianteAscentState2(next_position, next_log_prob), next_log_prob
    

    first_state = SparseCoordianteAscentState2(X, lp) # type: ignore
    first_states = jax.tree.map(lambda x: jax.lax.broadcast(x, (n_chains,)), first_state)
    
    keys = jax.random.split(seed, (n_iter * int(jax.lax.ceil(1. / p)), n_chains))
    last_state, result = jax.lax.scan(jax.vmap(step), first_states, keys)

    return last_state.position, last_state.log_prob, result