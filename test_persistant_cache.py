import jax
# jax.config.update("jax_compilation_cache_dir", "./jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
# jax.config.update("jax_explain_cache_misses", True)
# import jax.numpy as jnp
import jax.profiler

@jax.jit
def f(x):
    print(x)
    y = 0
    for i in range(1_000):
        y += x[i] + jax.random.normal(jax.random.key(i))
    return y

X = jax.random.normal(jax.random.key(0), (1_000,))
# print("X", X.device)
# y = f(X)
# print("y", y.device)
X_cpu = jax.device_get(X)
print("X", X_cpu.device)
y2 = f(X_cpu)
print("y", y2.device)
jax.profiler.save_device_memory_profile("memory.prof")