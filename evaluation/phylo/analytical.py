# phyppl/probabilistic-programming/blob/master/webppl/phyjs/index.js

import jax
import jax.numpy as jnp
from trees import Node, Leaf, Branch, count_leaves, ln_factorial

def exact_CRBD_loglikelihood(tree: Branch, lam: jax.Array, mu: jax.Array, rho: jax.Array) -> jax.Array:
    # Compute correction factor from oriented to labelled unoriented trees
    num_leaves = count_leaves(tree)
    corr_factor = (num_leaves - 1) * jnp.log(2.0) - ln_factorial(num_leaves)

    # Non-stalked tree, unconditional likelihood
    ln1 = (num_leaves - 2) * jax.lax.log(lam)
    ln2 = num_leaves * jax.lax.log(rho)
    ln3 = 2.0 * CRBD_ln_ghat(tree.age, lam, mu, rho)
    ln4 = CRBD_ln_likelihood(tree.left, lam, mu, rho) + CRBD_ln_likelihood(tree.right, lam, mu, rho)
    ln5 = -num_leaves * CRBD_ln_ghat(0, lam, mu, rho)
    
    return jax.lax.select(mu < lam, (corr_factor + ln1 + ln2 + ln3 + ln4 + ln5), -jnp.inf)


def CRBD_ln_ghat(t: float, lam: jax.Array, mu: jax.Array, rho: jax.Array) -> jax.Array:
    r = lam - mu
    eRT = jax.lax.exp(-r*t)
    f = jax.lax.log(lam - (lam - r/rho)*eRT)
    return( -r*t - 2.0*f )


def CRBD_ln_likelihood(tree: Node, lam: jax.Array, mu: jax.Array, rho: jax.Array):
    if isinstance(tree, Leaf):
        return 0
    else:
        assert isinstance(tree, Branch)
        lnLikeLeft  = CRBD_ln_likelihood(tree.left, lam, mu, rho)
        lnLikeRight = CRBD_ln_likelihood(tree.right, lam, mu, rho)
        return lnLikeLeft + lnLikeRight + CRBD_ln_ghat(tree.age, lam, mu, rho)

def CRBD_lnS(t: float, lam: jax.Array, mu: jax.Array, rho: jax.Array) -> jax.Array:
    r = lam - mu
    eRT = jax.lax.exp(-r*t)
    lnNum = jax.lax.log(r)
    lnDenom = jax.lax.log(lam - (lam - r/rho)*eRT)
    return (lnNum - lnDenom)

def CRBD_survivorship_bias(tree: Node, lam: jax.Array, mu: jax.Array, rho: jax.Array) -> jax.Array:
    return -2.0*CRBD_lnS(tree.age, lam, mu, rho)


def main(tree: Branch, lam: jax.Array, epsilon: jax.Array, rho: jax.Array):
    mu = epsilon * lam
    return exact_CRBD_loglikelihood(tree, lam, mu, rho) + CRBD_survivorship_bias(tree, lam, mu, rho)

# for lam in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]:
#     for epsilon in [0.0, 0.1, 0.5, 0.9]:
#         print(lam, epsilon, main(bisse_32, jnp.array(lam,float), jnp.array(epsilon,float), jnp.array(1.,float)))