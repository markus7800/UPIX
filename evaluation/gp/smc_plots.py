from gp_smc import *
import matplotlib.pyplot as plt
from dccxjax.infer.dcc.mc_dcc import MCDCCResult


def plot_smc_posterior(weighted_samples: SampleValues[Tuple[Trace,FloatArray]], n, title):
    xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))
    
    key = jax.random.key(0)
    
    _, weights = weighted_samples.get()
        
    samples = []
    for i in range(n):
        key, trace_select_key, sample_key = jax.random.split(key, 3)
        trace_ix = dist.Categorical(weights).sample(trace_select_key)
        trace, weight = weighted_samples.get_selection(trace_ix)
        
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
    
def plot_results(m: Model, result: MCDCCResult[Trace]):
    slp_weights = list(result.get_slp_weights().items())
    slp_weights.sort(key=lambda v: v[1])

    
    for i in range(min(len(slp_weights),5)):
        slp, weight = slp_weights[-(i+1)]
        weighted_samples = result.get_samples_for_slp(slp).unstack()
        plot_smc_posterior(weighted_samples, 100, slp.formatted() + f" {weight:.6f}")
        

    plt.figure()
    plt.scatter(xs, ys)
    plt.scatter(xs_val, ys_val)
    xs_pred = jnp.hstack((xs,jnp.linspace(1.,1.5,50)))

    n_posterior_samples = 1_000
    sample_key = jax.random.key(0)
    slp_weights_array = jnp.array([weight for _, weight in slp_weights])
    posterior_over_slps = dist.Categorical(slp_weights_array)

    samples = []
    for i in tqdm(range(n_posterior_samples), desc="Sample posterior"):
        sample_key, key1, key2, key3 = jax.random.split(sample_key, 4)
        slp_ix = posterior_over_slps.sample(key1)
        slp, _ = slp_weights[slp_ix]
        weighted_samples = result.get_samples_for_slp(slp).unstack()
        _, weights = weighted_samples.get()
        trace_ix = dist.Categorical(weights).sample(key2)
        trace, weight = weighted_samples.get_selection(trace_ix)
        
        k = get_gp_kernel(trace)
        noise = transform_param("noise", trace["noise"]) + 1e-5
        mvn = k.posterior_predictive(xs, ys, noise, xs_pred, noise)


        if i < 10:
            tqdm.write(f"sample from posterior: {slp.formatted()} noise={noise} with log_prob {m.log_prob(trace)}")

            plt.plot(xs_pred, mvn.numpyro_base.mean, color="black", alpha=0.1)
            q025, q975 = mvnormal_quantile(mvn, 0.025), mvnormal_quantile(mvn, 0.975)
            plt.fill_between(xs_pred, q025, q975, alpha=0.1, color="tab:blue")

        samples.append(mvn.sample(key3))
    # plt.show()


    samples = jnp.vstack(samples)
    q050 = jnp.median(samples, axis=0)
    q025 = jnp.quantile(samples, 0.025, axis=0)
    q975 = jnp.quantile(samples, 0.975, axis=0)

    plt.figure()
    plt.scatter(xs, ys)
    plt.scatter(xs_val, ys_val)
    plt.plot(xs_pred, q050, color="black")
    plt.fill_between(xs_pred, q025, q975, alpha=0.5, color="tab:blue")
    plt.show()