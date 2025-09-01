import pandas as pd
import pathlib
import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

path = pathlib.Path("experiments", "pedestrian", "scale")

results = []
for file in os.listdir(path):
    if file.endswith(".json"):
        with open(path.joinpath(file), "r") as f:
            jdict = json.load(f)
            df_dict = jdict["workload"] | jdict["timings"] | jdict["dcc_timings"] | jdict["pconfig"] | jdict["environment_info"]
            del df_dict["environ"]
            del df_dict["jax_environment"]
            results.append(df_dict)
            
            
df = pd.DataFrame(results)
df = df[["platform", "num_workers", "n_chains", "inference_time"]]
df = df.set_index(["platform", "num_workers", "n_chains"], verify_integrity=True)
df = df.sort_index()
df = df.reset_index()

path = pathlib.Path("experiments", "pedestrian", "comp")
npdhmc_results = []
for file in os.listdir(path):
    if file.startswith("npdhmc") and file.endswith(".json"):
        with open(path.joinpath(file), "r") as f:
            jdict = json.load(f)
            df_dict = jdict["workload"] | jdict["timings"] | jdict["environment_info"]

            npdhmc_results.append(df_dict)
            

npdhmc_df = pd.DataFrame(npdhmc_results)
npdhmc_df = npdhmc_df[["platform", "cpu_count", "n_chains", "inference_time"]]
npdhmc_df = npdhmc_df.set_index(["platform", "cpu_count", "n_chains"], verify_integrity=True)
npdhmc_df = npdhmc_df.sort_index()
npdhmc_df = npdhmc_df.reset_index()
print(npdhmc_df)

from matplotlib.ticker import LogLocator

fig, ax = plt.subplots()

plt.xscale("log")
ax.xaxis.set_major_locator(LogLocator(base=2.0, subs=None))
ax.xaxis.set_minor_locator(LogLocator(base=2.0, subs=[]))
xticks = [2**n for n in range(20+1)]
ax.set_xticks(xticks, [f"{x:,d}" for x in xticks], rotation=75)
ax.set_xlabel("number of MCMC chains")

plt.yscale("log")
# ax.set_ylim(bottom=10, top=300)
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
yticks = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
ax.set_yticks(yticks, [f"{y:,d}" for y in yticks])
ax.set_ylabel("Inference time [s]")

# colors = {
#     "cpu": "tab:blue",
#     "gpu": "tab:orange",
#     "npdhmc": "tab:green"
# }

def get_shade(color, i, n):
    i = i - 1
    rgb = np.array(mcolors.to_rgb(color))
    return (1 - i/(n-1)) * np.array([1,1,1]) + (i/(n-1)) * rgb

# M = ["H", "o", "s", "D"]
# M = [".", "^", "x", "*"]
M = ["o", "^", "s", "D"]

markers = {
    ("gpu", 1): M[0],
    ("gpu", 2): M[1],
    ("gpu", 4): M[2],
    ("gpu", 8): M[3],
    ("cpu", 8): M[0],
    ("cpu", 16): M[1],
    ("cpu", 32): M[2],
    ("cpu", 64): M[3],
    ("npdhmc", 8): M[0],
    ("npdhmc", 16): M[1],
    ("npdhmc", 32): M[2],
    ("npdhmc", 64): M[3],
}
colors = {
    ("gpu", 1): get_shade("tab:blue", 4, 10),
    ("gpu", 2): get_shade("tab:blue", 6, 10),
    ("gpu", 4): get_shade("tab:blue", 8, 10),
    ("gpu", 8): get_shade("tab:blue", 10, 10),
    ("cpu", 8): get_shade("tab:orange", 4, 10),
    ("cpu", 16): get_shade("tab:orange", 6, 10),
    ("cpu", 32): get_shade("tab:orange", 8, 10),
    ("cpu", 64): get_shade("tab:orange", 10, 10),
    ("npdhmc", 8): get_shade("tab:green", 10, 10),
    ("npdhmc", 16): get_shade("tab:green", 10, 10),
    ("npdhmc", 32): get_shade("tab:green", 10, 10),
    ("npdhmc", 64): get_shade("tab:green", 10, 10),
}
ax.grid(True)
ax.set_axisbelow(True)

for (platform, num_workers), group in df.groupby(["platform", "num_workers"]):
    plt.plot(group["n_chains"], group["inference_time"],
             label=f"{num_workers} {platform}", 
             c=colors[(platform, num_workers)],
             marker=markers[(platform, num_workers)],
    )
    
for _, r in npdhmc_df.iterrows():
    plt.scatter(r.n_chains, r.inference_time, label=f"NP-DHMC {r.cpu_count} cpu", c=colors[("npdhmc",r.cpu_count)], marker=markers[("npdhmc", r.cpu_count)])

plt.legend(ncol=3)
plt.title("Pedestrian Model")

plt.tight_layout()
plt.show()