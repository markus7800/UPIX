import jax
import jax.numpy as jnp
from typing import Optional, Sequence, List, Dict, Tuple, NamedTuple, Callable, Generic, TypeVar, Any, cast

# adapted from jax.example_libraries/optimizers.py

# learning rate schedules

__all__ = [
    "constant",
    "exponential_decay",
    "inverse_time_decay",
    "polynomial_decay",
    "piecewise_constant",
    "SGD",
    "Momentum",
    "Adagrad",
    "Adam"
]

Schedule = Callable[[int], float]


def constant(step_size) -> Schedule:
    def schedule(i):
        return step_size
    return schedule


def exponential_decay(step_size, decay_steps, decay_rate):
    def schedule(i):
        return step_size * decay_rate ** (i / decay_steps)
    return schedule


def inverse_time_decay(step_size, decay_steps, decay_rate, staircase=False):
    if staircase:
        def schedule(i):
            return step_size / (1 + decay_rate * jnp.floor(i / decay_steps))
    else:
        def schedule(i):
            return step_size / (1 + decay_rate * i / decay_steps)
    return schedule


def polynomial_decay(step_size, decay_steps, final_step_size, power=1.0):
    def schedule(step_num):
        step_num = jnp.minimum(step_num, decay_steps)
        step_mult = (1 - step_num / decay_steps) ** power
        return step_mult * (step_size - final_step_size) + final_step_size

    return schedule


def piecewise_constant(boundaries: Any, values: Any):
    boundaries = jnp.array(boundaries)
    values = jnp.array(values)
    if not boundaries.ndim == values.ndim == 1:
        raise ValueError("boundaries and values must be sequences")
    if not boundaries.shape[0] == values.shape[0] - 1:
        raise ValueError(
            "boundaries length must be one shorter than values length")

    def schedule(i):
        return values[jnp.sum(i > boundaries)]
    return schedule


def make_schedule(scalar_or_schedule: float | Schedule) -> Schedule:
    if callable(scalar_or_schedule):
        return scalar_or_schedule
    elif jnp.ndim(scalar_or_schedule) == 0:
        return constant(scalar_or_schedule)
    else:
        raise TypeError(type(scalar_or_schedule))


# we do not need pytrees / we operate on arrays
OPTIMIZER_STATE = TypeVar("OPTIMIZER_STATE")
OPTIMIZER_PARAMS = jax.Array
OPTIMIZER_UPDATES = jax.Array


class Optimizer(NamedTuple, Generic[OPTIMIZER_STATE]):
    init_fn: Callable[[OPTIMIZER_PARAMS], OPTIMIZER_STATE]
    update_fn: Callable[[int, OPTIMIZER_UPDATES,
                         OPTIMIZER_STATE], OPTIMIZER_STATE]
    get_params_fn: Callable[[OPTIMIZER_STATE], OPTIMIZER_PARAMS]


def SGD(step_size_or_schedule: float | Schedule) -> Optimizer:
    schedule = make_schedule(step_size_or_schedule)

    def init(x0: jax.Array) -> jax.Array:
        return x0

    def update(i: int, g: jax.Array, x: jax.Array) -> jax.Array:
        return x - schedule(i) * g

    def get_params(x):
        return x
    return Optimizer(init, update, get_params)


def Momentum(step_size_or_schedule: float | Schedule, mass: float):
    schedule = make_schedule(step_size_or_schedule)

    def init(x0: jax.Array) -> Tuple[jax.Array, jax.Array]:
        v0 = jnp.zeros_like(x0)
        return x0, v0

    def update(i: int, g: jax.Array, state: Tuple[jax.Array, jax.Array]) -> Tuple[jax.Array, jax.Array]:
        x, velocity = state
        velocity = mass * velocity + g
        x = x - schedule(i) * velocity
        return x, velocity

    def get_params(state: Tuple[jax.Array, jax.Array]) -> jax.Array:
        x, _ = state
        return x
    return Optimizer(init, update, get_params)


def Adagrad(step_size_or_schedule: float | Schedule, momentum=0.9) -> Optimizer:
    schedule = make_schedule(step_size_or_schedule)

    def init(x0: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        g_sq = jnp.zeros_like(x0)
        m = jnp.zeros_like(x0)
        return x0, g_sq, m

    def update(i: int, g: jax.Array, state: Tuple[jax.Array, jax.Array, jax.Array]) -> Tuple[jax.Array, jax.Array, jax.Array]:
        x, g_sq, m = state
        g_sq += jnp.square(g)
        g_sq_inv_sqrt = jnp.where(g_sq > 0, 1. / jnp.sqrt(g_sq), 0.0)
        m = (1. - momentum) * (g * g_sq_inv_sqrt) + momentum * m
        x = x - schedule(i) * m
        return x, g_sq, m

    def get_params(state: Tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        x, _, _ = state
        return x

    return Optimizer(init, update, get_params)


def Adam(step_size_or_schedule: float | Schedule, b1=0.9, b2=0.999, eps=1e-8) -> Optimizer:
    schedule = make_schedule(step_size_or_schedule)

    def init(x0: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0

    def update(i: int, g: jax.Array, state: Tuple[jax.Array, jax.Array, jax.Array]):
        x, m, v = state
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
        # Bias correction.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        x = x - schedule(i) * mhat / (jnp.sqrt(vhat) + eps)
        return x, m, v

    def get_params(state: Tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        x, _, _ = state
        return x
    return Optimizer(init, update, get_params)
