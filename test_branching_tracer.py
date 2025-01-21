import jax
from dccxjax import *
import jax._src.core as jax_core
import jax.numpy as jnp

def test_1(x: jax.Array):
    if x < 0:
        return -x
    else:
        return x
    
# jax.make_jaxpr(test_1)(1) # errors

# numpy methods are partially jitted, see jax/_src/numpy/ufuncs.py
def test_2(x: jax.Array):
    print("test_2", x)
    return jnp.sin(4*x)

# jax.jvp(test_2, (2.,), (1.,))

a = detect_branching(test_2)(1)

print("returns", a)

# test_2(1)



# detect_branching(test_1)(1)

def test_3(x: jax.Array):
    print("test_3", x)
    return jax.lax.sin(4*x)

print(jax.jvp(test_2, (2.,), (1.,)))

a = detect_branching(test_3)(jnp.array([1.]))
print("returns", a)
a = detect_branching(test_3)(jnp.array(1.))
print("returns", a)
a = detect_branching(test_3)(1.)
print("returns", a)