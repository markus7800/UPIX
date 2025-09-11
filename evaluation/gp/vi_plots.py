from gp_vi import *
from dccxjax.infer.dcc.vi_dcc import VIDCCResult


def plot_guide_posterior(g: Guide, n, title):
    xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))
    
    key = jax.random.key(0)
    posterior = Traces(g.sample(key, (n,)), n)
    
    samples = []
    for i in range(n):
        key, sample_key = jax.random.split(key)
        trace = posterior.get_ix(i)
        k = get_gp_kernel(trace)
        noise = transform_param("noise", trace["noise"]) + 1e-5
        mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)
        samples.append(mvn.sample(sample_key))

    samples = jnp.vstack(samples)
    m = jnp.mean(samples, axis=0)
    q025 = jnp.quantile(samples, 0.025, axis=0)
    q975 = jnp.quantile(samples, 0.975, axis=0)
    
    plt.figure()
    plt.scatter(xs, ys)
    plt.scatter(xs_val, ys_val)
    plt.plot(xs_pred, m, color="black")
    plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
    plt.title(title)
    
def plot_results(result: VIDCCResult):
    slp_weights = list(result.get_slp_weights().items())
    slp_weights.sort(key=lambda v: v[1])

    for i in range(min(len(slp_weights),5)):
        slp, weight = slp_weights[-(i+1)]
        print(slp.formatted(), weight)
        g = result.slp_guides[slp]
        
        plot_guide_posterior(g, 100, slp.formatted() + f" {weight:.6f}")
        
    plt.show()