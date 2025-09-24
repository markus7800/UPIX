from upix.backend import set_host_device_count
set_host_device_count(2)

import jax
import jax.experimental


def callback_fn(i, v):
    print(f"Iter {i} {v}")

def step(carry, key):
    iteration, s = carry
    _ = jax.lax.cond(
            (iteration % 10 == 0),
            lambda _: jax.experimental.io_callback(callback_fn, None, iteration, s),
            lambda _: None,
            operand=None,
        )
    # jax.experimental.io_callback(lambda i, v: print(f"Iter {i} {v}"), None, iteration, s)
    s = s + jax.random.normal(key)
    return (iteration + 1, s), None

def f(seed):
    return jax.lax.scan(step, (0, 0.), jax.random.split(seed, 100))[0]

print(f(jax.random.key(0)))

# print(jax.vmap(f)(jax.random.split(jax.random.key(0),2))) # not ok

print(jax.pmap(f)(jax.random.split(jax.random.key(0),2))) # ok


def body(iteration, carry):
    key, s = carry
    _ = jax.lax.cond(
            (iteration % 10 == 0),
            lambda _: jax.experimental.io_callback(callback_fn, None, iteration, s),
            lambda _: None,
            operand=None,
        )
    key1, key2 = jax.random.split(key)
    s = s + jax.random.normal(key1)
    return (key2, s)
def f2(seed):
    return jax.lax.fori_loop(0, 100, body, (seed, 0.0))[1]

print(f2(jax.random.key(0)))

# print(jax.vmap(f2)(jax.random.split(jax.random.key(0),2)))

print(jax.pmap(f2)(jax.random.split(jax.random.key(0),2))) # ok
