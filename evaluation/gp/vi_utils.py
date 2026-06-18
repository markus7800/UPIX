from gp_vi import *
from upix.infer.dcc.vi_dcc import VIDCCResult

def lppd_guide(g: Guide, xs, ys, xs_val, ys_val, n, seed):
    key = jax.random.key(seed)
    posterior = Traces(g.sample(key, (n,)), n)
    pell = 0.0
    lppd = -jnp.inf
    for i in range(n):
        trace = posterior.get_ix(i)
        k = get_gp_kernel(trace)
        noise = transform_param("noise", trace["noise"]) + 1e-5
        mvn = k.posterior_predictive(xs, ys, noise, xs_val, noise)
        lp = mvn.log_prob(ys_val)
        pell += lp
        lppd = jnp.logaddexp(lppd, lp)
    return pell / n, lppd - jnp.log(n)
    
    
def compute_lppd_old(result: VIDCCResult, xs, ys, xs_val, ys_val, n, seed):
    slp_weights = list(result.get_slp_weights().items())
    slp_weights.sort(key=lambda v: v[1])

    pell = 0.0 # posterior expected log-likelihood = sum (log p(y|theta_i)) / n
    lppd = -jnp.inf # log posterior predictive density = log (sum p(y|theta_i) / n)
    for i in range(len(slp_weights)):
        slp, weight = slp_weights[-(i+1)]
        g = result.slp_guides[slp]
        
        pell_g, lppd_g = lppd_guide(g, xs, ys, xs_val, ys_val, n, seed*1000+i)
        pell += pell_g * weight
        lppd = jnp.logaddexp(lppd, lppd_g + jnp.log(weight))
        
    return float(pell), float(lppd)

    
def compute_lppd(result: VIDCCResult, xs, ys, xs_val, ys_val, n, seed):
    slp_weights = list(result.get_slp_weights().items())
    weights = jnp.array([w for _, w in slp_weights],float)
    slp_dist = dist.Categorical(weights / weights.sum()) # is normalised in log-space, so be safe here

    pell = 0.0 # posterior expected log-likelihood = sum (log p(y|theta_i)) / n
    lppd = -jnp.inf # log posterior predictive density = log (sum p(y|theta_i) / n)
    
    rng_key = jax.random.key(seed)
    
    for _ in range(n):
        rng_key, slp_key, guide_key = jax.random.split(rng_key, 3)
        slp_ix = slp_dist.sample(slp_key)
        slp, _ = slp_weights[slp_ix]
        g = result.slp_guides[slp]
        trace = g.sample(guide_key)
        k = get_gp_kernel(trace)
        noise = transform_param("noise", trace["noise"]) + 1e-5
        mvn = k.posterior_predictive(xs, ys, noise, xs_val, noise)
        lp = mvn.log_prob(ys_val)
        pell += lp
        lppd = jnp.logaddexp(lppd, lp)
        
    return float(pell / n), float(lppd - jnp.log(n))
        

def save_results(args, result: VIDCCResult, vi_dcc_obj: VIDCC, timings: dict, pell: float, pell_std: Optional[float], lppd: float, lppd_std: Optional[float], folder: str):
    K = int(vi_dcc_obj.advi_n_runs) * int(vi_dcc_obj.advi_L)
    workload = {
        "K": K,
        "L": vi_dcc_obj.advi_L,
        "n_runs": vi_dcc_obj.advi_n_runs,
        "n_iter": vi_dcc_obj.advi_n_iter,
        "n_slps": len(result.get_slps()),
        "config": NODE_CONFIG.NAME,
        "seed": args.seed
    }

    result_metrics = {
        "result_str": result.sprint(sortkey="slp"),
        "pell": pell,
        "lppd": lppd
    }
    if pell_std is not None:
        result_metrics["pell_std"] = pell_std
    if lppd_std is not None:
        result_metrics["lppd_std"] = lppd_std
        
    json_result = {
        "workload": workload,
        "timings": timings,
        "dcc_timings": vi_dcc_obj.get_timings(),
        "result_metrics": result_metrics,
        "args": args.__dict__,
        "pconfig": vi_dcc_obj.pconfig.__dict__,
        "environment_info": get_environment_info()
    }
    
    if not args.no_save:
        prefix = f"K_{K:07d}_nruns_{vi_dcc_obj.advi_n_runs}_L_{vi_dcc_obj.advi_L}_nslps_{len(result.get_slps())}_niter_{vi_dcc_obj.advi_n_iter}_"
        write_json_result(json_result, "gp", "vi", folder, prefix=prefix)


    
def plot_guide_posterior(g: Guide, xs, ys, xs_val, ys_val, n, title):
    xs_pred = jnp.hstack((xs,jnp.linspace(xs_val[0], xs_val[0] + 4*(xs_val[-1]-xs_val[0]),50)))
    
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
    
def plot_results(result: VIDCCResult, xs, ys, xs_val, ys_val):
    slp_weights = list(result.get_slp_weights().items())
    slp_weights.sort(key=lambda v: v[1])

    for i in range(min(len(slp_weights),5)):
        slp, weight = slp_weights[-(i+1)]
        print(slp.formatted(), weight)
        g = result.slp_guides[slp]
        
        plot_guide_posterior(g, xs, ys, xs_val, ys_val, 100, slp.formatted() + f" {weight:.6f}")
        
    plt.show()