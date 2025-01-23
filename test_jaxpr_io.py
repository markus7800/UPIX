import jax
import jax.numpy as jnp

@jax.jit
def f1(x: jax.Array, y: jax.Array):
    return x + y
# f1(jnp.zeros((2,3)), jnp.zeros((2,3)))

@jax.jit
def f2(x: jax.Array):
    return x, x
# f2(jnp.zeros((2,3)))

@jax.jit
def f3(x: jax.Array, y: jax.Array):
    return x, y
# f3(jnp.zeros((2,3)), jnp.zeros((2,3)))


@jax.jit
def f4(d):
    x = d["x"]
    y = d["y"]
    return {"z": x + y}
f4({"x": jnp.zeros((2,3)), "y": jnp.zeros((2,3)), "a": {"b": jnp.zeros(2)}})