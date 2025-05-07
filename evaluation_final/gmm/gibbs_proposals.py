
from dccxjax import *
import jax
import jax.numpy as jnp
import dccxjax.distributions as dist

class WProposal(TraceProposal):
    def __init__(self, delta: float, K: int) -> None:
        self.delta = delta
        self.K = K

    def get_dirichlet(self, current: Trace):
        K = self.K + 1
        zs = current["zs"]
        counts = jnp.sum(zs.reshape(-1,1) == jnp.arange(0,K).reshape(1,-1), axis=0)
        d = dist.Dirichlet(counts + self.delta)
        return d

    def propose(self, rng_key: PRNGKey, current: Trace) -> Tuple[Trace,jax.Array]:
        d = self.get_dirichlet(current)
        proposed = d.sample(rng_key)
        lp = d.log_prob(proposed)
        return {"w": proposed}, lp
    
    def assess(self, current: Trace, proposed: Trace) -> jax.Array:
        d = self.get_dirichlet(current)
        return d.log_prob(proposed["w"])
    
class MusProposal(TraceProposal):
    def __init__(self, ys: jax.Array, kappa: float, xi: float, K: int) -> None:
        self.ys = ys
        self.kappa = kappa
        self.xi = xi
        self.K = K

    def get_gaussian(self, current: Trace):
        K = self.K + 1
        vars = current["vars"]
        zs = current["zs"]
        cluster_alloc_mat = zs.reshape(-1,1) == jnp.arange(0,K).reshape(1,-1)
        cluster_counts = jnp.sum(cluster_alloc_mat, axis=0)
        cluster_y_sum = jnp.sum(self.ys.reshape(-1,1) * cluster_alloc_mat, axis=0)
        
        return dist.Normal(
            jnp.where(cluster_counts > 0, (cluster_y_sum / vars + self.kappa * self.xi) / (cluster_counts/vars + self.kappa), self.xi),
            jnp.where(cluster_counts > 0, jnp.sqrt(1 / (cluster_counts / vars + self.kappa)), 1 / jnp.sqrt(self.kappa))
        )
        
    def propose(self, rng_key: PRNGKey, current: Trace) -> Tuple[Trace,jax.Array]:
        d = self.get_gaussian(current)
        proposed = d.sample(rng_key)
        lp = d.log_prob(proposed).sum()
        return {"mus": proposed}, lp
    
    def assess(self, current: Trace, proposed: Trace) -> jax.Array:
        d = self.get_gaussian(current)
        return d.log_prob(proposed["mus"]).sum()

class VarsProposal(TraceProposal):
    def __init__(self, ys: jax.Array, alpha: float, beta: float, K: int) -> None:
        self.ys = ys
        self.alpha = alpha
        self.beta = beta
        self.K = K

    def get_invgamma(self, current: Trace):
        K = self.K + 1
        mus = current["mus"]
        zs = current["zs"]
        cluster_alloc_mat = zs.reshape(-1,1) == jnp.arange(0,K).reshape(1,-1)
        cluster_counts = jnp.sum(cluster_alloc_mat, axis=0)
        cluster_y_devs = jnp.sum(((self.ys.reshape(-1,1) - mus.reshape(1,-1)) ** 2) * cluster_alloc_mat, axis=0)
        
        return dist.InverseGamma(
            jnp.where(cluster_counts > 0, self.alpha + cluster_counts / 2, self.alpha),
            jnp.where(cluster_counts > 0, self.beta + cluster_y_devs / 2, self.beta)
        )
        
    def propose(self, rng_key: PRNGKey, current: Trace) -> Tuple[Trace,jax.Array]:
        d = self.get_invgamma(current)
        proposed = d.sample(rng_key)
        lp = d.log_prob(proposed).sum()
        return {"vars": proposed}, lp
    
    def assess(self, current: Trace, proposed: Trace) -> jax.Array:
        d = self.get_invgamma(current)
        return d.log_prob(proposed["vars"]).sum()
    
class ZsProposal(TraceProposal):
    def __init__(self, ys: jax.Array) -> None:
        self.ys = ys

    def get_categorical(self, current: Trace) -> dist.Categorical | dist.CategoricalLogits:
        mus = current["mus"]
        vars = current["vars"]
        w = current["w"]
        
        # def get_cat_probs(y):
        #     p = jnp.exp(dist.Normal(mus, jnp.sqrt(vars)).log_prob(y)) * w
        #     p = jax.lax.select(p.sum() < 1e-32, jax.lax.full_like(w, 1.), p)
        #     return p / p.sum()
        
        # cat = dist.Categorical(jax.vmap(get_cat_probs)(self.ys))
        # return cat
    
        def get_cat_log_probs(y):
            # unnormalised is fine
            log_p = dist.Normal(mus, jnp.sqrt(vars)).log_prob(y) + jax.lax.log(w)
            return log_p
        
        cat = dist.CategoricalLogits(jax.vmap(get_cat_log_probs)(self.ys))
        return cat
        
        
        
    def propose(self, rng_key: PRNGKey, current: Trace) -> Tuple[Trace,jax.Array]:
        d = self.get_categorical(current)
        proposed = d.sample(rng_key)
        lp = d.log_prob(proposed).sum()
        return {"zs": proposed}, lp
    
    def assess(self, current: Trace, proposed: Trace) -> jax.Array:
        d = self.get_categorical(current)
        lp = d.log_prob(proposed["zs"])
        return lp.sum()