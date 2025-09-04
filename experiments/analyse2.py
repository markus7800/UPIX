import pandas as pd
import pathlib
import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("folder")
args = parser.parse_args()

PATH = pathlib.Path(args.folder)

METRIC_COL = "inference_time"
if PATH.name == "pedestrian":
    SCALE_COL = "n_chains"
    TITLE = "Pedestrian Model"
    X_LABEL = "Number of HMC chains"

elif PATH.parent.name == "gp" and PATH.name == "vi":
    SCALE_COL = "L"
    TITLE = "Gaussian Process Model"
    X_LABEL = "Number of samples per ADVI step L"
    
elif PATH.parent.name == "gp" and PATH.name == "smc":
    SCALE_COL = "n_particles"
    TITLE = "Gaussian Process Model"
    X_LABEL = "Number of SMC particles"
    
elif PATH.name == "gmm":
    SCALE_COL = "n_chains"
    TITLE = "Gaussian Mixture Model"
    X_LABEL = "Number of MH chains"
    
else:
    exit(1)

results = []
for dirpath, _, files in os.walk(PATH.joinpath("scale")):
    for file in files:
        if file.endswith(".json"):
            with open(pathlib.Path(dirpath, file), "r") as f:
                jdict = json.load(f)
                df_dict = jdict["workload"] | jdict["timings"] | jdict["dcc_timings"] | jdict["pconfig"] | jdict["environment_info"]
                del df_dict["environ"]
                del df_dict["jax_environment"]
                results.append(df_dict)
            
            
df = pd.DataFrame(results)
if PATH.name == "gmm":
    df = df[df["n_samples_per_chain"] == 2048]
    
df["gpu-brand"] = df["gpu-brand"].map(lambda x: x[0][len("GPU 0: "):])
    
df = df[["platform", "gpu-brand", "num_workers", SCALE_COL, "total_time", "inference_time", "jax_total_jit_time", "n_available_devices"]]
df = df.set_index(["platform", "gpu-brand", "num_workers", SCALE_COL], verify_integrity=True)
df = df.sort_index()
df = df.reset_index()

from matplotlib.ticker import LogLocator

fig, ax = plt.subplots()

plt.xscale("log")
ax.xaxis.set_major_locator(LogLocator(base=2.0, subs=None))
ax.xaxis.set_minor_locator(LogLocator(base=2.0, subs=[]))
xticks = [2**n for n in range(20+1)]
ax.set_xticks(xticks, [f"{x:,d}" for x in xticks], rotation=75)
ax.set_xlabel(X_LABEL)

plt.yscale("log")
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
yticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
ax.set_yticks(yticks, [f"{y:,d}" for y in yticks])
ax.set_ylabel(METRIC_COL + " [s]")
# ax.set_ylim(bottom=1, top=2500)

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
}

colors = {
    "NVIDIA L40S": "tab:blue",
    "NVIDIA A40": "tab:orange",
    "NVIDIA A100-SXM4-40GB": "tab:green",
}

def get_color(brand, n_devices):
    s = (np.log2(n_devices) + 2) * 2
    return get_shade(colors[brand], s, 10)
ax.grid(True)
ax.set_axisbelow(True)

series_count = 0

for (platform, gpu, n_devices), group in df.groupby(["platform", "gpu-brand", "num_workers"]):
    print(group)
    series_count += 1
    plt.plot(group[SCALE_COL], group[METRIC_COL],
             label=f"{n_devices} {platform} {gpu}", 
             c=get_color(gpu, n_devices),
             marker=markers[(platform, n_devices)],
    )
    
plt.legend(ncol= np.ceil(series_count / 4))
plt.title(TITLE)

plt.tight_layout()
plt.show()