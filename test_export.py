import jax
from typing import NamedTuple

import jax.export

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

print(f(jax.random.PRNGKey(1)))

exported = jax.export.export(f)(jax.random.PRNGKey(0))

serialized: bytearray = exported.serialize()
# print(serialized)

with open("serialised_jax_func.bin", "wb") as file:
    file.write(serialized)


with open("serialised_jax_func.bin", "rb") as file:
    assert file.read() == serialized

