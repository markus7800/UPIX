import jax
import jax._src.core as jax_core



x = jax_core.get_aval(1)
print(x)
print(getattr(x, "_rmul"))
