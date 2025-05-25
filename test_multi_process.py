
import jax.numpy as jnp
import jax
# import multiprocessing
import multiprocess

@jax.jit
def step(carry, any):
  val, key = carry
  key1, key2 = jax.random.split(key)
  val = val + jax.random.normal(key1)
  return (val,key2), None
  
@jax.jit
def f(seed):
  print(seed)
  return jax.lax.scan(step, (jnp.array(0.,float), seed), length=10**7)[0][0]

def worker_fn(seed):
  return f(seed)

class A:
  def __init__(self, seed) -> None:
    self.f = f
    self.seed = seed


def worker_fn_2(a: A):
  return a.f(a.seed)

if __name__ == "__main__":
  f(jax.random.PRNGKey(0))
  # with multiprocessing.get_context("spawn").Pool(2) as p:
  #   r = p.imap(worker_fn, [jax.random.PRNGKey(0), jax.random.PRNGKey(1)])
  #   print(list(r))
  with multiprocess.get_context("spawn").Pool(2) as p:
    r = p.imap(worker_fn_2, [A(jax.random.PRNGKey(0)), A(jax.random.PRNGKey(1))])
    print(list(r))

# if __name__ == "__main__":
# #   f(jax.random.PRNGKey(0))
#     with multiprocessing.get_context("spawn").Pool(2) as p:
#         r = p.imap(worker_fn, [jax.random.PRNGKey(0), jax.random.PRNGKey(1)])
#     # with multiprocess.Pool(2) as p:
#     #     r = p.imap(worker_fn_2, [A(jax.random.PRNGKey(0)), A(jax.random.PRNGKey(1))])
#     print(list(r))