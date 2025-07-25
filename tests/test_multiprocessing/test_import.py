import jax
from typing import NamedTuple

import jax.export

class B(NamedTuple):
    x: jax.Array
    y: jax.Array

jax.export.register_namedtuple_serialization(B, serialized_name="A")

with open("serialised_jax_func.bin", "rb") as file:
    serialized: bytes = file.read()

rehydrated_exp = jax.export.deserialize(bytearray(serialized))

print(rehydrated_exp.call(jax.random.key(1)))