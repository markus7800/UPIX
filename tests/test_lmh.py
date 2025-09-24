from upix.all import *
import upix.distributions as dist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

@model
def geometric(p: FloatArrayLike, obs: FloatArrayLike):
    b = True
    i = 0
    while b:
        b = sample(f"b_{i}", dist.Bernoulli(p))
        i += 1
    sample("x", dist.Normal(i, 1.0), observed=obs)


m = geometric(0.5, 5.)

def find_i_max(X: Trace):
    i = 0
    for addr in X.keys():
        i = max(i, int(addr[2:]))
    return i

rng_keys = jax.random.split(jax.random.key(0), 100_000)
result = [m.generate(rng_key) for rng_key in tqdm(rng_keys)]
r = jnp.array([find_i_max(X) for X, _ in result])
lps = jnp.array([lp - m.log_prior(X) for X, lp in result])

w = jnp.exp(lps - jax.scipy.special.logsumexp(lps))
for i in range(10):
    print(i, w[r == i].sum())

rng_key = jax.random.key(0)
rng_key, generate_key = jax.random.split(rng_key)
X, _ = m.generate(generate_key)
r = []
for _ in tqdm(range(100_000)):
    rng_key, step_key, accept_key = jax.random.split(rng_key, 3)
    X_proposed, acceptance_logprob = lmh(m, AllVariables(), X, step_key)
    if jax.lax.log(jax.random.uniform(accept_key)) < acceptance_logprob:
        X = X_proposed
    r.append(find_i_max(X))
r = jnp.array(r)

for i in range(10):
    print(i, jnp.mean(r == i))

# plt.hist(i, weights=w, bins=jnp.arange(0,10), density=True)
# plt.show()
