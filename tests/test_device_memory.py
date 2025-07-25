import jax
print(jax.devices('cpu'))
print(jax.devices('gpu'))

cpu = jax.devices('cpu')[0]
gpu = jax.devices('gpu')[0]

rng_key = jax.random.key(0)
arrs = []
for i in range(100):
    print(i)
    rng_key, sample_key = jax.random.split(rng_key)
    arr = jax.random.normal(sample_key, (1000*1000, 25))
    arr.block_until_ready()
    arrs.append(arr) # should be 100MB
    if i % 10 == 0:
        for j in range(i):
            # arrs[j] = jax.device_put(arrs[j], device=cpu)
            arrs[j] = jax.device_get(arrs[j]) # transfer to host


def f(rng_key) -> jax.Array:
    print(rng_key)
    return jax.random.normal(rng_key, (1000*1000, 25))

f_jitted = jax.jit(f)

res_1 = f_jitted(jax.device_put(rng_key, device=cpu))
print(res_1.device)
res_2 = f_jitted(jax.device_put(rng_key, device=gpu))
print(res_2.device)

f_jitted_2 = jax.jit(f, device=gpu)

res_1 = f_jitted_2(jax.device_put(rng_key, device=cpu))
print(res_1.device)
res_2 = f_jitted_2(jax.device_put(rng_key, device=gpu))
print(res_2.device)