

import jax
import jax.numpy as jnp


def f(seed, N):
    def step(carry, key):
        s, i = carry
        minval = jax.lax.max(jnp.array(-1,float), -s) # stay >= 0
        maxval = jax.lax.min(jnp.array(1,float), i-s-1) # in the next steps i can substract from s at most i-1 units
        u = jax.random.uniform(key, minval=minval, maxval=maxval)
        return (s + u, i-1), u
    s, us = jax.lax.scan(step, (jnp.array(0.,float),N), jax.random.split(seed, N))
    return s, us

# N = jnp.array(5, int)
# for seed in range(10):
#     s, us = f(jax.random.key(seed), N)
#     cs = jax.lax.cumsum(us)
#     assert (cs >= 0).all()
#     assert cs[-1] == 0.
#     print(s, us, cs)
    
    
def g(seed, xs):
    max_sub = jax.lax.cumsum((jnp.maximum(-1,-(xs + 1)))[::-1])[::-1]
    max_sub = -jax.lax.concatenate((max_sub[1:], jnp.array([0.],float)), 0)
    def step(carry, data):
        key, x, ms = data
        s, d = carry
        minval = jax.lax.max(jnp.array(-1,float), -s)
        minval = jax.lax.max(minval, -(x + 1))
        maxval = jax.lax.min(jnp.array(1,float), ms-s)
        maxval = jax.lax.min(maxval, 1 - x)
        u = jax.random.uniform(key, minval=minval, maxval=maxval)
        return (s + u, d + jax.lax.abs(u)), u
    s, us = jax.lax.scan(step, (jnp.array(0.,float), jnp.array(0.,float)), (jax.random.split(seed, xs.size), xs, max_sub))
    return s, us,

# for seed in range(100):
#     xs = jax.random.uniform(jax.random.key(seed), (5,), minval=-1, maxval=1)
#     # xs = jnp.zeros(5)
#     # print(xs, jnp.maximum(-1,-(xs + 1)))
#     # max_sub = jax.lax.cumsum((jnp.maximum(-1,-(xs + 1)))[::-1])[::-1]
#     # print(xs, jax.lax.concatenate((max_sub[1:], jnp.array([0.],float)), 0))
#     s, us = g(jax.random.key(seed+10), xs)
#     assert (xs + us <= 1).all()
#     assert (xs + us >= -1).all()
#     cs = jax.lax.cumsum(us)
#     assert (cs >= 0).all()
#     assert cs[-1] == 0.
#     print(s, us, cs)
    
    

def h(seed, start, stop, N):
    def step(carry, key):
        s, i = carry
        minval = jax.lax.max(jnp.array(-1,float), jax.lax.select(i == 1, -s + stop, -s + 1e-5))
        maxval = jax.lax.min(jnp.array(1,float), (i-1)*(1-1e-5) - s + stop)
        u = jax.random.uniform(key, minval=minval, maxval=maxval)
        return (s + u, i-1), u
    s, us = jax.lax.scan(step, (start, N), jax.random.split(seed, N))
    return s[0], us

# N = jnp.array(5, int)
# for seed in range(10):
#     key1, key2, key3 = jax.random.split(jax.random.key(seed), 3)
#     start = jax.random.uniform(key1, minval=0, maxval=3)
#     stop = -0.1
#     s, us = h(key3, start, stop, N)
#     cs = start + jax.lax.cumsum(us)
#     print(start, stop, s, us, cs, jnp.sum(jnp.abs(us)))
#     assert (cs[:-1] > 0).all()
#     assert cs[-1] < 0
    
    
def generate_init_pedestrian(keys, N):
    @jax.vmap
    def _init(key):
        key1, key2, key3 = jax.random.split(key,3)
        distance = 0.1 * jax.random.normal(key1) + 1.1
        start = jax.random.uniform(key2, minval=0, maxval=jnp.minimum(N,3))
        _, steps = h(key3, start, -0.1, N)
        scale = jnp.minimum(distance / jnp.sum(jnp.abs(steps)), 1)
        return start * scale, steps * scale
    return _init(keys)


starts, steps = generate_init_pedestrian(jax.random.split(jax.random.key(0), 10), 3)
print("starts")
print(starts)
print("steps", steps.shape)
print(steps)
print("stops")
print(starts.reshape(-1,1) + jnp.cumsum(steps, axis=1))
print("distance")
print(jnp.sum(jnp.abs(steps), axis=1))

