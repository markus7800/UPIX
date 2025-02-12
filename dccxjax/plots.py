from .inference import DCC_Result
import matplotlib.pyplot as plt
import jax

def plot_histogram(result: DCC_Result, address: str):
    samples, weights, undef_prob = result.get_samples_for_address(address, unstack_chains=True)

    fig, axs = plt.subplots(1,2,sharey="all",width_ratios=[0.9,0.1])
    plt.subplots_adjust(wspace=0, hspace=0)
    if samples is not None and weights is not None:
        axs[0].hist(samples, weights=weights, density=True, bins=100, alpha=0.5)
        kde = jax.scipy.stats.gaussian_kde(samples, weights=weights)
        xs = jax.numpy.linspace(samples.min(), samples.max(), 200)
        axs[0].plot(xs, kde(xs), color="tab:blue")
    axs[1].bar(["undef"],[undef_prob])
    axs[1].margins(0.5, 0)
    fig.suptitle(f"Posterior of \"{address}\"")

    return fig


def plot_trace(result: DCC_Result, address: str):
    slp_samples = {slp: slp_samples[address] for slp, slp_samples in result.samples.items() if address in slp_samples}
    Z = sum(z for _, z in result.Zs.items())

    fig, axs = plt.subplots(len(slp_samples),2,sharex="col", sharey="col", width_ratios=[0.5,1.])
    
    for i, (slp,samples) in enumerate(slp_samples.items()):
        assert len(samples.shape) == 2 # multi-dim variables not plottable
        n_chains = samples.shape[1]
        
        _axs = axs[i] if len(slp_samples) > 1 else axs
        w = result.Zs[slp] / Z
        _axs[0].set_title(f"w = {w:.4f}")
        _axs[0].set_ylabel(slp.branching_decisions.to_human_readable())

        for j in range(n_chains):
            chain = samples[:,j]
            kde = jax.scipy.stats.gaussian_kde(chain)
            xs = jax.numpy.linspace(chain.min(), chain.max(), 200)
            density = kde(xs)
            density = density.at[0].set(0.)
            density = density.at[-1].set(0.)
            _axs[0].plot(xs, density, color="tab:blue", alpha=max(0.1,1/n_chains))
            _axs[1].plot(chain, alpha=max(0.1,1/n_chains))
        # _axs[0].set_ylim([0.,None])

    fig.suptitle(f"Markov chains for \"{address}\"")
    plt.tight_layout()

    return fig