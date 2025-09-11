from run_scale_vi import *
from dccxjax.infer import ADVI
from vi_plots import plot_guide_posterior

if __name__ == "__main__":
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    m.set_equivalence_map(equivalence_map)
    
    vi_dcc_obj = VIConfig2(m, verbose=2,
        advi_n_iter = args.n_iter,
        advi_L=args.L,
        advi_optimizer=Adam(0.005),
        elbo_estimate_n_samples=100,
        parallelisation = get_parallelisation_config(args),
        disable_progress=args.no_progress,
        slp_max_n_leaf = 3,
    )
    
    slp_str = "((1Per + 1RQ) * 1Poly)"
    # slp_str = "((1Per * 1Poly) + 1RQ)"
    # slp_str = "(1Poly + 1RQ)"
    
    slp : SLP = None # type: ignore
    slps: List[SLP] = []
    vi_dcc_obj.initialise_active_slps(slps, [], jax.random.key(0))
    for i, _slp in enumerate(slps):
        if _slp.formatted() == slp_str:
            print("Ding ding!", i)
            slp = _slp
            
            
            
    fig, ax = plt.subplots()
    for seed in range(10):
        advi = ADVI(slp, vi_dcc_obj.get_guide(slp), Adam(0.005), 8, pconfig=vi_dcc_obj.pconfig,
                    show_progress=True,
                    shared_progressbar=None)
        last_state, advi_elbo = advi.run(jax.random.key(seed), n_iter=2_325)
        
        guide = advi.get_updated_guide(last_state)
        ax.plot(advi_elbo, alpha=0.5, c="tab:blue")
    plt.show()
    exit()
        
        
        
    fig, ax = plt.subplots()
    
    for L in [2**e for e in range(0,7)]:
        
        print(f"{L=}")
        advi = ADVI(slp, vi_dcc_obj.get_guide(slp), Adam(0.005), L, pconfig=vi_dcc_obj.pconfig,
                    show_progress=True,
                    shared_progressbar=None)
        last_state, advi_elbo = advi.run(jax.random.key(0), n_iter=2_000)
        
        guide = advi.get_updated_guide(last_state)
        Xs, lqs = guide.sample_and_log_prob(jax.random.key(0), shape=(10_000,))
        lps = jax.vmap(slp.log_prob)(Xs)
        true_elbo = jnp.mean(lps - lqs)
        ax.plot(advi_elbo, label=f"L={L} ELBO={true_elbo.item():.2f}", alpha=0.5)
        
        plot_guide_posterior(guide, 100, f"L={L} ELBO={true_elbo.item():.2f}")
        
    ax.legend()
    plt.show()
    
    
# uv run -p python3.13 --with=pandas evaluation/gp/smc_scale_viz.py sequential smap_local 256 0 0 -host_device_count=8