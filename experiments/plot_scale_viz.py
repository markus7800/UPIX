import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import pathlib
import sys

folder = sys.argv[1]

file = "viz_ped_mcmc_scale_data.pkl"
file = "viz_gmm_mcmc_scale_data.pkl"
# file = ""
if file:
    with open(pathlib.Path(folder, file), "rb") as f:
        res = pickle.load(f)

        n_chains_to_W1_distance, n_chains_to_infty_distance = res
        
        fig, ax = plt.subplots()
        plt.title("W1 distance")
        ax.boxplot(n_chains_to_W1_distance.values()) # type: ignore
        ax.set_xticklabels(n_chains_to_W1_distance.keys(), rotation=75)
        plt.tight_layout()

        fig, ax = plt.subplots()
        plt.title("Infty distance")
        ax.boxplot(n_chains_to_infty_distance.values()) # type: ignore
        ax.set_xticklabels(n_chains_to_infty_distance.keys(), rotation=75)
        plt.tight_layout()
        plt.show()
        
file = "viz_gp_smc_particle_scale_data.pkl"
file = "viz_gp_vi_elbo_scale_data.pkl"
file = ""
if file:
    with open(pathlib.Path(folder, file), "rb") as f:
        res = pickle.load(f)

        fig, ax = plt.subplots()
        ax.boxplot(res.values()) # type: ignore
        ax.set_xticklabels(res.keys(), rotation=75)
        plt.show()
        