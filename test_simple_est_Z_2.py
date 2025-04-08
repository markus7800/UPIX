import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt


w = jnp.array([0.1,0.2,0.3,0.3,0.1])
X = dist.CategoricalProbs(w)
C1 = jnp.array([0,1])
C2 = jnp.array([2,3,4])

X_sample = X.sample(jax.random.PRNGKey(0), (10_000_000,))
print(jnp.array([jnp.mean(X_sample == x) for x in X.enumerate_support()]))

def log_P_X_constrained(C):
    def _log_P_X_constrained(x):
        return jax.lax.select(jnp.isin(x, C), X.log_prob(x), -jnp.inf)
    return _log_P_X_constrained


Z1 = w[C1].sum()
Z2 = w[C2].sum()
print(f"{Z1=} {Z2=}")

log_is_weights = jax.vmap(log_P_X_constrained(C1))(X_sample) - X.log_prob(X_sample)
print("Z1_est", jnp.exp(jax.scipy.special.logsumexp(log_is_weights)) / X_sample.size)

log_is_weights = jax.vmap(log_P_X_constrained(C2))(X_sample) - X.log_prob(X_sample)
print("Z2_est", jnp.exp(jax.scipy.special.logsumexp(log_is_weights)) / X_sample.size)

key1, key2 = jax.random.split(jax.random.PRNGKey(0),2)

N = 10_000_000

w1 = jnp.exp(jnp.array([log_P_X_constrained(C1)(x) for x in X.enumerate_support()]))
w1 = w1 / w1.sum()
print(f"{w1=}")
X1 = dist.CategoricalProbs(w1)
X1_sample = X1.sample(key1, (N,)) # sample from correct sub-posterior
lp1 = jax.vmap(log_P_X_constrained(C1))(X1_sample)

w2 = jnp.exp(jnp.array([log_P_X_constrained(C2)(x) for x in X.enumerate_support()]))
w2 = w2 / w2.sum()
print(f"{w2=}")
X2 = dist.CategoricalProbs(w2)
X2_sample = X2.sample(key2, (N,)) # sample from correct sub-posterior
lp2 = jax.vmap(log_P_X_constrained(C2))(X2_sample)

print(f"w1 * Z1 + w2 * Z2 = {w1 * Z1 + w2 * Z2}")


mean_p_1 = jnp.mean(jnp.exp(lp1))
mean_p_2 = jnp.mean(jnp.exp(lp2))

w1_sq_sum = jnp.sum(jnp.exp(jnp.array([log_P_X_constrained(C1)(x) for x in X.enumerate_support()])) ** 2)
w2_sq_sum = jnp.sum(jnp.exp(jnp.array([log_P_X_constrained(C2)(x) for x in X.enumerate_support()])) ** 2)

Z_ratio = (Z2 / Z1)

Z_ratio_2 = (mean_p_1 * w1_sq_sum) / (mean_p_2 * w2_sq_sum)

print(f"Z2/Z1 = {Z_ratio.item()} vs {Z_ratio.item()}")
print(f"p1/p2 = {(mean_p_1 / mean_p_2).item()}")


print(mean_p_1 / (mean_p_1 + mean_p_2), mean_p_2 / (mean_p_1 + mean_p_2))




def step(carry, rng_key):
    c_key, r_key, a_key = jax.random.split(rng_key,3)
    current_col, current_row, current_x, current_lp = carry
    col = jax.random.randint(c_key, (), 0, N)
    row = jax.random.randint(r_key, (), 0, 2)

    # lp = jax.lax.select(row == 0, lp1[col], lp2[col]) # wrong, gives ratios [mean_p_1, mean_p_2] / (mean_p_1 + mean_p_2)
    # lp = jax.lax.select(row == 0, X.log_prob(X1_sample[col]), X.log_prob(X2_sample[col])) # equivalent
    # accept = jax.lax.log(jax.random.uniform(a_key)) < (lp - current_lp)

    # this is correct:
    x = jax.lax.select(row == 0, X1_sample[col], X2_sample[col])
    # x = jax.lax.select(row == 0, X1.sample(c_key), X2.sample(c_key)) # equivalent
    lp = X.log_prob(x)
    q = jnp.log(0.5 * jnp.exp(X1.log_prob(current_x)) + 0.5 * jnp.exp(X2.log_prob(current_x))) - jnp.log(0.5 * jnp.exp(X1.log_prob(x)) + 0.5 * jnp.exp(X2.log_prob(x)))
    accept = jax.lax.log(jax.random.uniform(a_key)) < (lp - current_lp + q)
    
    next_col, next_row, next_x, next_lp = jax.lax.cond(accept, lambda _: (col, row, x, lp), lambda _: (current_col, current_row, current_x, current_lp), None)
    return (next_col, next_row, next_x, next_lp), next_row

_, rows = jax.lax.scan(step, (0,0,0,-jnp.inf), jax.random.split(jax.random.PRNGKey(0), 1_000_000))
print(jnp.array([jnp.mean(rows == i).item() for i in range(2)]))