from run_scale_vi import *
from dccxjax.infer import ADVI
from vi_plots import plot_guide_posterior
from dccxjax.parallelisation import VectorisationType

if __name__ == "__main__":
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    m.set_equivalence_map(equivalence_map)
    
    args.n_slps = 128
    
    vi_dcc_obj = VIConfig2(m, verbose=2,
        advi_optimizer=Adam(0.005),
        elbo_estimate_n_samples=100,
        parallelisation = get_parallelisation_config(args),
        disable_progress=args.no_progress,
        slp_max_n_leaf = 3,
    )
    
    slp_str = "((1Per + 1RQ) * 1Poly)"
    
    slp : SLP = None # type: ignore
    slps: List[SLP] = []
    vi_dcc_obj.initialise_active_slps(slps, [], jax.random.key(0))
    for i, _slp in enumerate(slps):
        if _slp.formatted() == slp_str:
            print("Ding ding!", i)
            slp = _slp
            
    assert vi_dcc_obj.pconfig.vectorisation in (VectorisationType.GlobalSMAP, VectorisationType.GlobalVMAP, VectorisationType.LocalSMAP, VectorisationType.LocalVMAP)
    
    # if vi_dcc_obj.pconfig.vectorisation == VectorisationType.LocalVMAP:
    #     vi_dcc_obj.pconfig.vectorisation = VectorisationType.GlobalVMAP
    # if vi_dcc_obj.pconfig.vectorisation == VectorisationType.LocalSMAP:
    #     vi_dcc_obj.pconfig.vectorisation = VectorisationType.GlobalSMAP
            
    # fig, ax = plt.subplots()
    # advi = ADVI(slp, vi_dcc_obj.get_guide(slp), Adam(0.005), 1, 20, pconfig=vi_dcc_obj.pconfig,
    #             show_progress=True,
    #             shared_progressbar=None)
    # last_state, advi_elbo = advi.run(jax.random.key(0), n_iter=2_325)
    # print(advi_elbo[-1,:])
    # print(advi_elbo.shape)
    # ax.plot(advi_elbo, alpha=0.5, c="tab:blue")
    # plt.show()

    # if vi_dcc_obj.pconfig.vectorisation == VectorisationType.GlobalVMAP:
    #     vi_dcc_obj.pconfig.vectorisation = VectorisationType.LocalVMAP
    # if vi_dcc_obj.pconfig.vectorisation == VectorisationType.GlobalSMAP:
    #     vi_dcc_obj.pconfig.vectorisation = VectorisationType.LocalSMAP
        
    # fig, ax = plt.subplots()
    # for seed in range(20):
    #     advi = ADVI(slp, vi_dcc_obj.get_guide(slp), Adam(0.005), 20, 1, pconfig=vi_dcc_obj.pconfig,
    #                 show_progress=True,
    #                 shared_progressbar=None)
    #     last_state, advi_elbo = advi.run(jax.random.key(seed), n_iter=2_325)
        
    #     best_run = int(jnp.argmax(advi_elbo[-1,:]).item()) if advi.n_runs > 1 else None
    #     ax.plot(advi_elbo, alpha=0.5, c="tab:blue")
    # plt.show()
    
    if vi_dcc_obj.pconfig.vectorisation == VectorisationType.LocalVMAP:
        vi_dcc_obj.pconfig.vectorisation = VectorisationType.GlobalVMAP
    if vi_dcc_obj.pconfig.vectorisation == VectorisationType.LocalSMAP:
        vi_dcc_obj.pconfig.vectorisation = VectorisationType.GlobalSMAP
        
    K_to_elbos: Dict[str,jax.Array] = dict()
    for K in [2**e for e in range(0,13+1)]:
        print(f"{K=}")
        elbos = []
        for seed in range(10):
            advi = ADVI(slp, vi_dcc_obj.get_guide(slp), Adam(0.005), 1, K, pconfig=vi_dcc_obj.pconfig,
                        show_progress=True,
                        shared_progressbar=None)
            last_state, advi_elbo = advi.run(jax.random.key(seed), n_iter=1_000)
            
            best_run = int(jnp.argmax(advi_elbo[-1,:]).item()) if advi.n_runs > 1 else None
            guide = advi.get_updated_guide(last_state, best_run)
            Xs, lqs = guide.sample_and_log_prob(jax.random.key(seed), shape=(10_000,))
            lps = jax.vmap(slp.log_prob)(Xs)
            elbo = jnp.mean(lps - lqs)
            elbos.append(elbo)
            
            if seed == 0:
                plot_guide_posterior(guide, 100, f"K={K} ELBO={elbo.item():.2f}")
                

        elbos = jnp.vstack(elbos).reshape(-1)
        print(f"{K=} elbo est: {elbos.mean().item():.4f} +/- {elbos.std().item()}")
        K_to_elbos[f"{K:_}"] = elbos
    
    plt.show()
    
    fig, ax = plt.subplots()
    ax.boxplot(K_to_elbos.values()) # type: ignore
    ax.set_xticklabels(K_to_elbos.keys())
    plt.show()
    