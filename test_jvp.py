import jax
from dccxjax import *
import jax._src.core as jax_core
import jax.numpy as jnp

# def test_1(x: jax.Array):
#     return jax.lax.sin(x)

# a = deriv(test_1)(0.)
# print("returns", a)


def test_2(x: jax.Array, y: jax.Array):
    return jax.lax.sin(x + 2*y)

a = grad(test_2, (0,1))(1., 2.)
print("returns", a)

# a = mygrad(test_2)((1., 2.))
# print("returns", a)


# a = jax.grad(test_2, (0,1))(1., 2.)
# print("returns", a)

# jax.grad(test_1)(0.)