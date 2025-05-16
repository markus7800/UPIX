import jax

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
  # Run the operations to be profiled
  key = jax.random.key(0)
  x = jax.random.normal(key, (25000, 25000))
  y = x @ x @ x
  y.block_until_ready()