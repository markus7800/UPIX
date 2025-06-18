from dccxjax.infer.dcc.mc_dcc import MCDCCResult
import matplotlib.pyplot as plt
import jax
from typing import Optional

__all__ = [
    "plot_histogram",
    "plot_histogram_by_slp",
    "plot_trace",
]

def plot_histogram(result: MCDCCResult, address: str):
    weighted_samples, undef_prob = result.get_samples_for_address(address)

    fig, axs = plt.subplots(1,2,sharey="all",width_ratios=[0.9,0.1])
    plt.subplots_adjust(wspace=0, hspace=0)
    if weighted_samples is not None:
        samples, weights = weighted_samples.unstack().get()
        axs[0].hist(samples, weights=weights, density=True, bins=100, alpha=0.5)
        kde = jax.scipy.stats.gaussian_kde(samples, weights=weights)
        xs = jax.numpy.linspace(samples.min(), samples.max(), 200)
        axs[0].plot(xs, kde(xs), color="tab:blue")
    # axs[1].yaxis.set_label_position("right")
    # axs[1].yaxis.tick_right()
    axs[1].bar(["undef"],[undef_prob])
    axs[1].margins(0.5, 0)
    fig.suptitle(f"Posterior of \"{address}\"")

    plt.tight_layout()

    return fig

def plot_histogram_by_slp(result: MCDCCResult, address: str, N: Optional[int] = None):

    slps = result.get_slps_where_address_exists(address)
    slps = sorted(slps, key=lambda slp: slp.sort_key())
    if N is not None:
        slps = slps[:N]

    fig, axs = plt.subplots(len(slps)+1, 2, sharex="col", width_ratios=[0.9,0.1], figsize=(6.4, 2.*(len(slps)+1)), layout="constrained")
    Z = jax.lax.exp(result.get_log_weight_normaliser())

    weighted_samples, undef_prob = result.get_samples_for_address(address)
    assert weighted_samples is not None
    samples, weights = weighted_samples.unstack().get()
    _ax = axs[-1]
    _ax[0].hist(samples, weights=weights, density=True, bins=100, alpha=0.5)
    kde = jax.scipy.stats.gaussian_kde(samples, weights=weights)
    xs = jax.numpy.linspace(samples.min(), samples.max(), 200)
    _ax[0].plot(xs, kde(xs), color="tab:blue")
    _ax[1].yaxis.set_label_position("right")
    Z_combined = sum(jax.lax.exp(result.slp_log_weights[slp]) for slp in slps)
    _ax[1].set_ylabel(f"sum(Z)={Z_combined:.4g}")
    # _ax[1].yaxis.tick_right()
    _ax[0].set_ylabel("Combined")
    _ax[1].bar(["prob"],[Z_combined / Z])
    _ax[1].margins(0.5, 0)
    _ax[1].set_ylim((0.,1.))
    

    for i, slp in enumerate(slps):
        samples, weights = result.get_samples_for_address_and_slp(address, slp).unstack().get()
        assert len(samples.shape) == 1 # multi-dim variables not plottable

        _ax = axs[i]

        _ax[0].hist(samples, density=True, bins=100, alpha=0.5)#, weights=weights)
        kde = jax.scipy.stats.gaussian_kde(samples)#, weights=weights)
        xs = jax.numpy.linspace(samples.min(), samples.max(), 200)
        _ax[0].plot(xs, kde(xs), color="tab:blue")
        _ax[1].bar(["prob"],[jax.lax.exp(result.slp_log_weights[slp]) / Z])
        _ax[1].yaxis.set_label_position("right")
        # _ax[1].yaxis.tick_right()
        _ax[1].set_ylabel(f"Z={jax.lax.exp(result.slp_log_weights[slp]):.4g}")
        
        _ax[0].set_ylabel(slp.formatted())
        _ax[1].set_ylim((0.,1.))

    
    # plt.tight_layout()

    fig.suptitle(f"Posterior of \"{address} per SLP\"")

    return fig


def plot_trace(result: MCDCCResult, address: str):
    slps = result.get_slps_where_address_exists(address)
    slps = sorted(slps, key=lambda slp: slp.sort_key())
    Z = result.get_log_weight_normaliser()

    fig, axs = plt.subplots(len(slps), 2, sharex="col", sharey="col", width_ratios=[0.5,1.], figsize=(6.4, 2*len(slps)), layout="constrained")
    
    for i, slp in enumerate(slps):
        address_results = result.get_samples_for_address_and_slp(address, slp)
        n_chains = address_results.n_chains()
        
        _axs = axs[i] if len(slps) > 1 else axs
        w = jax.lax.exp(result.slp_log_weights[slp] - Z)
        _axs[0].set_title(f"w = {w:.4f}")
        _axs[0].set_ylabel(slp.formatted())

        for j in range(n_chains):
            chain, _ = address_results.get_chains(j).get()
            assert len(chain.shape) == 1 # multi dim traces plots not supported yet
            kde = jax.scipy.stats.gaussian_kde(chain)
            xs = jax.numpy.linspace(chain.min(), chain.max(), 200)
            density = kde(xs)
            density = density.at[0].set(0.)
            density = density.at[-1].set(0.)
            _axs[0].plot(xs, density, color="tab:blue", alpha=max(0.1,1/n_chains))
            _axs[1].plot(chain, alpha=max(0.1,1/n_chains))
        # _axs[0].set_ylim([0.,None])

    fig.suptitle(f"Markov chains for \"{address}\"")
    # plt.tight_layout()

    return fig