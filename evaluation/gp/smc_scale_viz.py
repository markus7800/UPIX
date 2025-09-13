from run_scale_smc import *
from dccxjax.infer import SMC, ReweightingType, StratifiedResampling, ResampleType, ResampleTime
from smc_plots import plot_smc_posterior

if __name__ == "__main__":
    m = gaussian_process(xs, ys)
    m.set_slp_formatter(lambda slp: str(get_gp_kernel(slp.decision_representative)))
    m.set_slp_sort_key(lambda slp: get_gp_kernel(slp.decision_representative).size())
    m.set_equivalence_map(equivalence_map)
    
    args.n_slps = 128
    smc_dcc_obj = SMCDCCConfig2(m, verbose=2,
        smc_collect_inference_info=True,
        parallelisation = get_parallelisation_config(args),
        disable_progress=args.no_progress,
        slp_max_n_leaf=3,
    )
    
    slp_str = "(GamExp + (Lin * Per))"
    
    slp : SLP = None # type: ignore
    slps: List[SLP] = []
    smc_dcc_obj.initialise_active_slps(slps, [], jax.random.key(0))
    for i, _slp in enumerate(slps):
        if _slp.formatted() == slp_str:
            print("Ding ding!", i)
            slp = _slp
        
    
    n_particles_to_log_Zs: Dict[str,jax.Array] = dict()
    repetitions = 10
    
    for smc_n_particles in [2**e for e in range(0,15+1)]:
        log_Zs = []
        for seed in range(repetitions):
            
            smc_dcc_obj.smc_n_particles = smc_n_particles
            smc = SMC(
                slp,
                smc_dcc_obj.smc_n_particles,
                smc_dcc_obj.get_SMC_tempering_schedule(slp),
                smc_dcc_obj.get_SMC_data_annealing_schedule(slp),
                ReweightingType.BootstrapStaticPrior,
                StratifiedResampling(ResampleType.Adaptive, ResampleTime.BeforeMove),
                smc_dcc_obj.get_SMC_rejuvination_kernel(slp),
                smc_dcc_obj.smc_rejuvination_attempts,
                pconfig=smc_dcc_obj.pconfig,
                collect_inference_info=False,
                show_progress=True,
                shared_progressbar=None
            )
            prior_key, smc_key = jax.random.split(jax.random.key(seed))
            init_positions, init_log_prob = smc_dcc_obj.produce_samples_from_path_prior(slp, prior_key)
            last_state, ess = smc.run(smc_key, init_positions, init_log_prob)
            log_Z = jax.scipy.special.logsumexp(last_state.log_particle_weights)
            # print(f"log_Z={log_Z.item():.4f}")
            log_Zs.append(log_Z)
            
            # if seed == 0:
            #     weights = jax.lax.exp(last_state.log_particle_weights - log_Z)
            #     weights = weights / weights.sum()
            #     weighted_samples = SampleValues((last_state.particles, weights), smc_n_particles)
            #     plot_smc_posterior(weighted_samples, 100, f"n_particles={smc_n_particles} log_Z={log_Z.item():.4f}")
        
        log_Zs = jnp.vstack(log_Zs).reshape(-1)
        print(f"{smc_n_particles=} log_Z est: {log_Zs.mean().item():.4f} +/- {log_Zs.std().item()}")
        n_particles_to_log_Zs[f"{smc_n_particles:_}"] = log_Zs
    # plt.show()
    
    fig, ax = plt.subplots()
    ax.boxplot(n_particles_to_log_Zs.values()) # type: ignore
    ax.set_xticklabels(n_particles_to_log_Zs.keys())
    plt.show()
    