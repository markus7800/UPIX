import jax, jax.numpy as jnp

def test_psum(x):
    local = jnp.sum(x)
    idx = jax.lax.axis_index("i")
    jax.debug.print("device {}/{}/{} local sum = {}", jax.local_device_count(), jax.process_index(), idx, local)
    global_sum = jax.lax.psum(local, 'i')
    return local, global_sum

x = jnp.ones((jax.local_device_count(), 1024))
local, global_sum = jax.pmap(test_psum,axis_name="i")(x)
print("local:", local)
print("global_sum:", global_sum)