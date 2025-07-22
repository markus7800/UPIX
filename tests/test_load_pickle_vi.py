import pickle
import jax
import jax.export

with open("tmp2.pickle", "rb") as f:
    work = pickle.load(f)
    jax_serialised_fn, args = work

    jax_fn = jax.export.deserialize(jax_serialised_fn)
    out = (jax_fn.call)(*args) # this will always compile

    print(out)