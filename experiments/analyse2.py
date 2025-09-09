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

from functools import reduce
def df_from_json_dir(dir, keep_info_subdicts: list[str], discard_names: list[str]):
    results = []
    for dirpath, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".json"):
                with open(pathlib.Path(dirpath, file), "r") as f:
                    jdict = json.load(f)
                    df_dict = reduce(lambda x, y: x | y, map(lambda x: jdict[x], keep_info_subdicts))
                    for name in discard_names:
                        del df_dict[name]
                    results.append(df_dict)
    return pd.DataFrame(results)

df = df_from_json_dir(PATH.joinpath("scale"), ["workload", "timings", "dcc_timings", "pconfig", "environment_info"], ["environ", "jax_environment"])
if PATH.name == "gmm":
    df = df[df["n_samples_per_chain"] == 2048]
    
df["kind"] = df["gpu-brand"].map(lambda x: x[0][len("GPU 0: "):])
df.loc[df["platform"] == "cpu", "kind"] = "CPU"
    
df = df[["platform", "kind", "num_workers", SCALE_COL, "total_time", "inference_time", "jax_total_jit_time", "n_available_devices"]]
df = df.set_index(["platform", "kind", "num_workers", SCALE_COL], verify_integrity=True)
df = df.sort_index()
df = df.reset_index()


# df["METRIC"] = df["total_time"] - df["jax_total_jit_time"]
# METRIC_NAME = "wall time without compilation"

df["METRIC"] = df["total_time"]
METRIC_NAME = "wall time"

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
ax.set_ylabel(METRIC_NAME + " [s]")
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
    "CPU": "tab:red",
    "COMP": "tab:purple"
}

def get_color(kind, n_devices):
    if kind == "CPU":
        s = (np.log2(n_devices) - 1) * 2 # maps [8,16,32,64] to [4,6,8,10]
    else:
        s = (np.log2(n_devices) + 2) * 2 # maps [1,2,4,8] to [4,6,8,10]
    return get_shade(colors[kind], s, 10)
ax.grid(True)
ax.set_axisbelow(True)

series_count = 0

for (platform, kind, n_devices), group in df.groupby(["platform", "kind", "num_workers"]):
    print(group)
    series_count += 1
    plt.plot(group[SCALE_COL], group["METRIC"],
            #  label=f"{n_devices} {platform} {kind}", 
             c=get_color(kind, n_devices),
             marker=markers[(platform, n_devices)],
    )
    

if PATH.name == "pedestrian":
    COMP_NAME = "NPDHMC"
    comp_df = df_from_json_dir(PATH.joinpath("nonparametric"), ["workload", "timings", "environment_info"], [])
    comp_df = comp_df[["platform", "cpu_count", SCALE_COL, "inference_time"]]
    comp_df["METRIC"] = comp_df["inference_time"]
    comp_df = comp_df.rename(columns={"cpu_count": "num_workers"})

elif PATH.parent.name == "gp" and PATH.name == "vi":
    COMP_NAME = "SDVI"
    comp_df = df_from_json_dir(PATH.joinpath("..", "sdvi"), ["workload", "timings", "environment_info"], [])
    comp_df = comp_df[["platform", "cpu_count", SCALE_COL, "inference_time"]]
    comp_df["METRIC"] = comp_df["inference_time"]
    comp_df = comp_df.rename(columns={"cpu_count": "num_workers"})
    
elif PATH.parent.name == "gp" and PATH.name == "smc":
    COMP_NAME = "AutoGP"
    comp_df = df_from_json_dir(PATH.joinpath("..", "autogp"), ["workload", "timings", "environment_info"], [])
    comp_df = comp_df[["platform", "cpu_count", SCALE_COL, "wall_time", "inference_time", "compilation_time"]]
    comp_df["METRIC"] = comp_df["wall_time"]
    comp_df = comp_df.rename(columns={"cpu_count": "num_workers"})
    
elif PATH.name == "gmm":
    COMP_NAME = "Gen RJMCMC"
    comp_df = df_from_json_dir(PATH.joinpath("rjmcmc"), ["workload", "timings", "environment_info"], [])
    comp_df = comp_df[["platform", "cpu_count", SCALE_COL, "wall_time", "inference_time", "compilation_time"]]
    comp_df["METRIC"] = comp_df["wall_time"]
    comp_df = comp_df.rename(columns={"cpu_count": "num_workers"})
    
else:
    assert(False)
    
comp_df["kind"] = "COMP"
comp_df = comp_df.set_index(["platform", "kind", "num_workers", SCALE_COL], verify_integrity=True)
comp_df = comp_df.sort_index()
comp_df = comp_df.reset_index()

for (platform, kind, n_devices), group in comp_df.groupby(["platform", "kind", "num_workers"]):
    series_count += 1
    plt.plot(group[SCALE_COL], group["METRIC"],
            # label=f"{n_devices} {platform} {kind}", 
            c=get_color(kind, n_devices),
            marker=markers[(platform, n_devices)],
    )
    
from matplotlib.lines import Line2D

legend_elements = []
for kind, color in colors.items():
    label = COMP_NAME if kind == "COMP" else kind
    legend_elements.append(Line2D([0], [0], color=color, lw=4, label=label))
for i in range(4):
    n_cpu = 2**(i+3)
    n_gpu = 2**(i)
    legend_elements.append(Line2D([0],[0], marker=M[i], color="black", label=f"{n_gpu} GPUs / {n_cpu} CPUs"))
    

ax.legend(handles=legend_elements)

# plt.legend(ncol= np.ceil(series_count / 8), loc='upper left')
plt.title(TITLE)

plt.tight_layout()
plt.show()