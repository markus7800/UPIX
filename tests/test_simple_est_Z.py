import jax
import jax.numpy as jnp
import numpyro.distributions as dist

d = dist.Normal(0,1)

T, N = 4, 10
x = d.sample(jax.random.key(0), (T, N))

lps = d.log_prob(x)

Zs = 1/jnp.array([1.,2.,3.,5.])
lps = lps + jnp.log(Zs).reshape(-1,1)

# print(lp)

Z_est = jnp.exp(jax.scipy.special.logsumexp(lps,axis=1))
Z_est = Z_est / Z_est.sum()

print(Zs / Zs.sum())
print(Z_est)

def step(carry, rng_key):
    c_key, r_key, a_key = jax.random.split(rng_key,3)
    current_col, current_row, current_lp = carry
    col = jax.random.randint(c_key, (), 0, N)
    row = jax.random.randint(r_key, (), 0, T)
    lp = lps[row, col]
    accept = jax.lax.log(jax.random.uniform(a_key)) < (lp - current_lp)
    next_col, next_row, next_lp = jax.lax.cond(accept, lambda _: (col, row, lp), lambda _: (current_col, current_row, current_lp), None)
    return (next_col, next_row, next_lp), next_row

_, rows = jax.lax.scan(step, (0,0,-jnp.inf), jax.random.split(jax.random.key(0), 1_000_000))
print(jnp.array([jnp.mean(rows == i).item() for i in range(T)]))