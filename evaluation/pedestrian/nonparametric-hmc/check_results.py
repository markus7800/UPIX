import pickle

from eval_utils import *

# file_template = "samples_produced/walk_model{i}__count1000_eps0.1_leapfrogsteps50.pickle"
# method = "hmc"

file_template = "lookahead_samples/walk_model_{i}__count1000_eps0.1_L5_alpha0.1_K2.pickle"
config = (5, 0.1, 2)
method = ("npladhmc-persistent", config)

runs = []
num_chains = 10
for i in range(num_chains):
    with open(file_template.format(i=i), "rb") as f:
        runs.append(pickle.load(f))

# print(runs)
if method == "hmc":
    thinned_runs = thin_runs([method], runs)
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
gt_xs = np.load("../gt_xs.npy")
gt_cdf = np.load("../gt_cdf.npy")
gt_pdf = np.load("../gt_pdf.npy")

hmc_values = np.array(values[method])

cdf_est = []
for x in gt_xs:
    cdf_est.append(np.mean(hmc_values < x))
cdf_est = np.array(cdf_est)

W1_distance = np.trapezoid(np.abs(cdf_est - gt_cdf)) # wasserstein distance
infty_distance = np.max(np.abs(cdf_est - gt_cdf))
title = f"W1 = {W1_distance.item():.4g}, L_inf = {infty_distance.item():.4g}"
print(title)
plt.plot(gt_xs, gt_cdf, label="ground truth")
plt.plot(gt_xs, cdf_est, label="estimated cdf")
plt.title(title)
plt.legend()
plt.show()

# Plot the data
import pandas
import seaborn as sns

data = (
    [("ours", v) for v in values[method]]
)
x_label = "starting point"
dataframe = pandas.DataFrame(data, columns=["method", x_label])
print(dataframe)
plot = sns.displot(
    data=dataframe,
    x=x_label,
    hue="method",
    kind="kde",
    common_norm=False,
    facet_kws={"legend_out": False},
    palette=palette,
    aspect=1,
    height=4,
)
plot.set_ylabels(label="posterior density")
plt.plot(gt_xs, gt_pdf, color="tab:orange", label="ground truth")
plt.show()

