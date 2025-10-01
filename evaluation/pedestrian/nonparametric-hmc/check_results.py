import pickle

from eval_utils import *

file_template = "samples_produced/walk_model{i}__count1000_eps0.1_leapfrogsteps50.pickle"
method = "hmc"

# file_template = "lookahead_samples/walk_model_{i}__count1000_eps0.1_L5_alpha0.1_K2.pickle"
# config = (5, 0.1, 2)
# method = ("npladhmc-persistent", config)

runs = []
num_chains = 8
for i in range(num_chains):
    with open(file_template.format(i=i), "rb") as f:
        runs.append(pickle.load(f))
        print(len(runs[i]["hmc"]["samples"]))

# print(runs)
if method == "hmc":
    thinned_runs = thin_runs([method], runs) # does not thin hmc samples, but restructures
    chains = collect_chains([method], thinned_runs)
    values = collect_values([method], thinned_runs)
    print_running_time([method], runs, thinned_runs)

else:
    values = {}
    chains = {}

    thinned_runs = thin_runs_2(runs, burnin=0)
    chains.update(collect_chains_2(thinned_runs, config=config))
    values.update(collect_values_2(thinned_runs, config=config))
    print_running_time_2(runs, thinned_runs)

import matplotlib.pyplot as plt
import numpy as np
gt_xs = np.load("../gt_xs-100.npy")
gt_cdf = np.load("../gt_cdf-100-1_000_000_000_000.npy")
gt_pdf = np.load("../gt_pdf_est-100-1_000_000_000_000.npy")

hmc_values = np.array(values[method])
print(f"{hmc_values.shape=}")

cdf_est = []
for x in gt_xs:
    cdf_est.append(np.mean(hmc_values < x))
cdf_est = np.array(cdf_est)

W1_distance = np.trapz(np.abs(cdf_est - gt_cdf), gt_xs) # wasserstein distance
infty_distance = np.max(np.abs(cdf_est - gt_cdf))
title = f"W1 = {W1_distance.item():.4g}, L_inf = {infty_distance.item():.4g}"
print(title)
plt.plot(gt_xs, gt_cdf, label="ground truth")
plt.plot(gt_xs, cdf_est, label="estimated cdf")
# plt.title(title)
plt.legend()
# plt.show()

import scipy
plt.figure(figsize=(5,2.5))
plt.hist(hmc_values, density=True, bins=100, alpha=0.5)
plt.plot(gt_xs, gt_pdf, label="ground truth")
kde = scipy.stats.gaussian_kde(hmc_values)
plt.plot(gt_xs, kde(gt_xs), color="tab:blue", label="NP-DHMC")
plt.ylim(0,1.4)
leg = plt.legend()
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_facecolor('none')
plt.tight_layout()
plt.savefig("result_nonparametric-hmc.pdf")
# plt.show()
