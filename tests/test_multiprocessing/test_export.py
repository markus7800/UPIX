import jax
from typing import NamedTuple

import jax.export
import jax.flatten_util

class A(NamedTuple):
    x: jax.Array
    y: jax.Array

jax.export.register_namedtuple_serialization(A, serialized_name="A")

@jax.jit
def f(rng_key: jax.Array):
    key1, key2 = jax.random.split(rng_key)
    x = jax.random.normal(key1)
    y = jax.random.normal(key2)
    return A(x, y)

@jax.jit
def _f(rng_key: jax.Array):
    out = f(rng_key)
    out, unravel_fn = jax.flatten_util.ravel_pytree(out)
    return out

print(_f(jax.random.key(1)))

exported = jax.export.export(_f)(jax.random.key(0))

serialized: bytearray = exported.serialize()
# print(serialized)

with open("serialised_jax_func.bin", "wb") as file:
    file.write(serialized)


with open("serialised_jax_func.bin", "rb") as file:
    assert file.read() == serialized

