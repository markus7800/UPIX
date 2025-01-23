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
# a = detect_branching(test_1)(1)
# print("returns", a)
# exit()

# numpy methods are partially jitted, see jax/_src/numpy/ufuncs.py
def test_2(x: jax.Array):
    print("test_2 x =", x)
    y = 4*x
    # y = jax.lax.sin(x)
    print("test_2 y =", type(y), y)
    return jnp.sin(y)
    # return jax.lax.cos(y)

# jax.jvp(test_2, (2.,), (1.,))


# a = detect_branching(test_2)(2.)
# print("returns", a)

@jax.jit
def scale_dict(x: jax.Array):
    return {"x": x * 2}

@jax.jit
def scale_within_dict(d: dict):
    x = d["x"]
    return {"x": x * 3}

def test_2_1(x: jax.Array):
    print("test_2 x =", x)
    y = scale_dict(x)
    print("test_2 y =", type(y), y)
    z = scale_within_dict(y)
    print("test_2 z =", type(z), z)
    return jnp.sin(z["x"])
    # return jax.lax.cos(y)

# a = detect_branching(test_2_1)(2.)
# print("returns", a)
# exit()

# @jax.jit
def scale_within_dict(d: dict):
    x = d["x"]
    return {"x": x * 3}

def test_2_2(d: dict):
    print("test_2 d =", type(d), d)
    z = scale_within_dict(d)
    print("test_2 z =", type(z), z)
    return jnp.sin(z["x"])
    # return jax.lax.cos(y)

# a = detect_branching(test_2_2)({"x": 2}) # dict is not a valid JAX type
# print("returns", a)
# exit()

@jax.jit
def scale_within_dict(d: dict):
    x = d["x"]
    return {"x": x * 3}

def test_2_3(x: jax.Array):
    d = {"x": x}
    print("test_2 d =", type(d), d)
    z = scale_within_dict(d)
    print("test_2 z =", type(z), z)
    return {"o": jnp.sin(z["x"])}
    # return jax.lax.cos(y)

a = detect_branching(test_2_3)(2)
print("returns", a)
exit()

# a = grad(test_2_2)({"x": 2})
# print("returns", a)
# exit()

# test_2(1)



# detect_branching(test_1)(1)

def test_3(x: jax.Array):
    return jax.lax.sin(x)

# print(jax.jvp(test_2, (2.,), (1.,)))

# a = detect_branching(test_3)(jnp.array([1.]))
# print("returns", a)
# a = detect_branching(test_3)(jnp.array(1.))
# print("returns", a)
# a = detect_branching(test_3)(1.)
# print("returns", a)
# a = detect_branching(jax.jit(test_3))(1.)
# print("returns", a)
# a = jax.jit(detect_branching(test_3))(1.)
# print("returns", a)


# a = jax.grad(test_3)(1.)
# print("returns", a)
# print("\n\n")
# a = jax.jit(jax.grad(test_3))(1.)
# print("returns", a)
# print("\n\n")
# a = jax.grad(jax.jit(test_3))(1.)
# print("returns", a)



a = detect_branching(test_3)(1.)
print("returns", a)
print("\n\n")
a = detect_branching(jax.grad(test_3))(1.)
print("returns", a)
print("\n\n")
a = jax.grad(detect_branching(test_3))(1.)
print("returns", a)