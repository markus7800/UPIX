import jax
from typing import Dict, Optional, List, Callable
from dccxjax.core import SLP, Model, sample_from_prior, slp_from_decision_representative
from ..types import Trace, PRNGKey
from dataclasses import dataclass
from .mcmc import _unstack_chains, n_samples_for_stacked_chains, InferenceRegime, get_inference_regime_mcmc_step_for_slp, get_initial_inference_state, unstack_chains
from .estimate_Z import estimate_Z_for_SLP_from_mcmc, estimate_Z_for_SLP_from_prior
from time import time
from copy import deepcopy

__all__ = [
    "DCC_Config",
    "dcc"
]

@dataclass
class DCC_Config:
    n_samples_from_prior: int
    n_chains: int
    collect_intermediate_chain_states: bool
    n_samples_per_chain: int
    n_samples_for_Z_est: int


class DCC_Result:
    def __init__(self, intermediate_states: bool) -> None:
        self.intermediate_states = intermediate_states
        self.samples: Dict[SLP, Trace] = dict()
        # samples have shape (n_samples, n_chain, dim(var))
        self.Zs: Dict[SLP, jax.Array] = dict()
    def add_samples(self, slp: SLP, samples: Trace, Z: Optional[jax.Array]):

        #  shape of trace entry for var:
        #  (n_samples_per_chain, n_chain, dim(var)) if multi_chain=True and intermediate_states=True
        #  (n_chain, dim(var)) if multi_chain=True and intermediate_states=False
        # if dim(var) == 1 dimension of trace entry is ommited, e.g. (n_samples_per_chain, n_chain) in the first case and () in the last case
        if not self.intermediate_states:
            samples = jax.tree_map(lambda x: jax.lax.broadcast(x, (1,)), samples)
        if slp not in self.samples:
            self.samples[slp] = samples
            assert Z is not None
            self.Zs[slp] = Z
        else:
            # samples have shape (n_samples_per_chain, n_chain, dim(var)) or (n_samples_per_chain, dim(var)) where n_samples_per_chain can be 1
            prev_samples = self.samples[slp]
            self.samples[slp] = jax.tree_map(lambda x, y: jax.lax.concatenate((x, y), 0), prev_samples, samples)
            if Z is not None:
                self.Zs[slp] = Z

    def get_samples_for_address_and_slp(self, address: str, slp: SLP, unstack_chains: bool = True):
        Z_slp = self.Zs[slp]
        Z = sum(z for _, z in self.Zs.items())
        samples = self.samples[slp]
        assert address in samples
        samples_for_address = samples[address]
        weigths_for_address = jax.lax.full_like(samples_for_address, Z_slp / Z)

        if unstack_chains:
            samples_for_address = _unstack_chains(samples_for_address)
            weigths_for_address = _unstack_chains(weigths_for_address)

        return samples_for_address, weigths_for_address

    def get_samples_for_address(self, address: str, unstack_chains: bool = True):
        undef_prob = 0.
        Z = sum(z for _, z in self.Zs.items())
        samples_for_address: Optional[jax.Array] = None
        weigths_for_address: Optional[jax.Array] = None
        for slp, samples in self.samples.items():
            Z_slp = self.Zs[slp]
            if address not in samples:
                undef_prob += Z_slp / Z
            else:
                weights = jax.lax.full_like(samples[address], Z_slp / Z)
                if samples_for_address is None:
                    samples_for_address = samples[address]
                    weigths_for_address = weights
                else:
                    assert weigths_for_address is not None
                    samples_for_address = jax.lax.concatenate((samples_for_address, samples[address]), 0)
                    weigths_for_address = jax.lax.concatenate((weigths_for_address, weights), 0)

        if unstack_chains and samples_for_address is not None and weigths_for_address is not None:
            samples_for_address = _unstack_chains(samples_for_address)
            weigths_for_address = _unstack_chains(weigths_for_address)
        
        return samples_for_address, weigths_for_address, undef_prob
    
    def n_samples(self):
        n = 0
        for _, samples in self.samples.items():
            n += n_samples_for_stacked_chains(samples)
        return n


def dcc(model: Model, regime_factory: Callable[[SLP], InferenceRegime], rng_key: PRNGKey, config: DCC_Config):
    all_slps: List[SLP] = []
    active_slps: List[SLP] = []
    proposed_slps: Dict[SLP,int] = dict()
    slp_to_mcmc_step = dict()
    slp_to_number_of_mcmc_runs: Dict[SLP,int] = dict()

    for _ in range(config.n_samples_from_prior):
        rng_key, key = jax.random.split(rng_key)
        X = sample_from_prior(model, key)
        slp = slp_from_decision_representative(model, X)

        if all(slp.path_indicator(X) == 0 for slp in active_slps):
            active_slps.append(slp)
            slp_to_mcmc_step[slp] = get_inference_regime_mcmc_step_for_slp(slp, regime_factory(slp), config.n_chains, config.collect_intermediate_chain_states)

    combined_result = DCC_Result(intermediate_states=config.collect_intermediate_chain_states)
    
    t0 = time()
    while True:
        did_mcmc = False
        for slp in active_slps:
            if slp_to_number_of_mcmc_runs.get(slp, 0) > 0:
                continue
            slp_to_number_of_mcmc_runs[slp] = slp_to_number_of_mcmc_runs.get(slp, 0) + 1
            did_mcmc = True

            mcmc_step = slp_to_mcmc_step[slp]
            print(f"Run mcmc for {slp.short_repr()}", slp.formatted())
            
            init = get_initial_inference_state(slp, config.n_chains)
            rng_key, key1, key2, key3 = jax.random.split(rng_key, 4)

            mcmc_keys = jax.random.split(key1, config.n_samples_per_chain)
            last_state, all_positions = jax.lax.scan(mcmc_step, init, mcmc_keys)
            last_positions = last_state.position
            # shape of entries of all_positions is (n_samples_per_chain, n_chains, dim(var))
            # shape of entries of last_positions is (n_chains, dim(var))

            Z, ess, _ = estimate_Z_for_SLP_from_prior(slp, config.n_samples_for_Z_est, key2)
            print(f"... estimated {Z=} with {ess=}")
            combined_result.add_samples(slp, all_positions if config.collect_intermediate_chain_states else last_positions, Z)

            Z2, _, _ = estimate_Z_for_SLP_from_mcmc(slp, 1.0, config.n_samples_for_Z_est // (config.n_samples_per_chain * config.n_chains) + 1, key2, unstack_chains(all_positions))
            print(f"{Z=} vs {Z2=}")

            # propose_new_slps_from_last_positions(slp, last_positions, active_slps, proposed_slps, config.n_chains, key3, 1.)
            # add_to_active: List[SLP] = []
            # for proposed_slp, count in proposed_slps.items():
            #     if count > config.n_chains // 10:
            #         add_to_active.append(proposed_slp)
            #         print(f"Add to active: slp {proposed_slp.short_repr()}", proposed_slp.formatted())
            # for proposed_slp in add_to_active:
            #     active_slps.append(proposed_slp)
            #     slp_to_mcmc_step[proposed_slp] = get_inference_regime_mcmc_step_for_slp(proposed_slp, deepcopy(regime), config.n_chains, config.collect_intermediate_chain_states)
            #     del proposed_slps[proposed_slp]


        if not did_mcmc:
            break

    t1 = time()
    t = t1 - t0
    n_samples = combined_result.n_samples()
    print("DCC: collected", n_samples, f"samples in {t:.3f}s ({n_samples / t / 1000:.3f}K/s)")

    # slp_desc = [slp.branching_decisions.to_human_readable().splitlines()[-1] for slp in active_slps]
    # for desc in sorted(slp_desc):
    #     print(desc)


    return combined_result

