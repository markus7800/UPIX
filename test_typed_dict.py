
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from dccxjax import *
from typing import TypedDict

class CarryStats(TypedDict, total=False):
    position: Trace
    unconstrained_position: Trace
    log_prob: FloatArray
    unconstrained_log_prob: FloatArray


class CarryStatsNoDict:
    def __init__(self, position: Trace) -> None:
        self.position: Trace = position


stats = CarryStats(position={"X": jnp.array(1.,float)}, log_prob=jnp.array(0.,float))
# stats["log_prob"]
x, unravel_fn = ravel_pytree(stats)
print(x)
x, unravel_fn = ravel_pytree(CarryStatsNoDict({"X": jnp.array(1.,float)}))
print(x)