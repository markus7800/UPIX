import pandas as pd
import pathlib
import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import argparse
from functools import reduce
import pickle
from matplotlib.ticker import LogLocator
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("folder")

args = parser.parse_args()

folder = args.folder


def df_from_json_dir(dir, keep_info_subdicts: list[str], discard_names: list[str]):
    results = []
    for dirpath, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".json"):
                # print(dirpath, file)
                with open(pathlib.Path(dirpath, file), "r") as f:
                    jdict = json.load(f)
                    df_dict = reduce(lambda x, y: x | y, map(lambda x: jdict[x], keep_info_subdicts))
                    for name in discard_names:
                        del df_dict[name]
                    results.append(df_dict)
    return pd.DataFrame(results)

plt.rcParams['text.usetex'] = True

gridspec = {
    "right": 0.63,
}
fig, axs = plt.subplots(1, 1, figsize=(5,3), gridspec_kw=gridspec)
# axs[2,0].set_visible(False)
# axs[2,1].set_visible(False)


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
    "CPU": "tab:red",
    "COMP": "tab:green"
}


linestyles = {
    "NVIDIA L40S": "solid",
    "NVIDIA A40": "dashed",
    "CPU": (0, (1,1)), # "densely dotted",
    "COMP": "dashdot"
}

def get_shade(color, i, n):
    i = i - 1
    rgb = np.array(mcolors.to_rgb(color))
    return (1 - i/(n-1)) * np.array([1,1,1]) + (i/(n-1)) * rgb

def get_color(kind, n_devices):
    if kind == "CPU":
        s = (np.log2(n_devices) - 1) * 2 # maps [8,16,32,64] to [4,6,8,10]
    else:
        s = (np.log2(n_devices) + 2) * 2 # maps [1,2,4,8] to [4,6,8,10]
    return colors[kind] # get_shade(colors[kind], s, 10)

device_legend_elements = []
for i in range(4):
    n_cpu = 2**(i+3)
    n_gpu = 2**(i)
    device_legend_elements.append(Line2D([0],[0], marker=M[i], lw=0, color="black", label=f"{n_gpu} {"GPUs" if n_gpu > 1 else "GPU"} / {n_cpu} CPUs"))

for kind, color in colors.items():
    if kind == "COMP":
        label = "Reference on CPU"
        continue
    elif kind == "CPU":
        label = "UPIX 2x EPYC 9355"
    else:
        label = "UPIX " + kind
    device_legend_elements.append(Line2D([0], [0], color=color, linestyle=linestyles[kind], lw=2, label=label))

# device_legend_elements = [device_legend_elements[i] for i in [0,4,1,5,2,6,3,7]]

for i, (model_path, SCALE_COL, comp_path, COMP_NAME, comp_time) in enumerate([
    ("pedestrian", "n_chains", "pedestrian/nonparametric", "NP-DHMC", "inference_time"),
    # ("gmm", "n_chains", "gmm/rjmcmc", "Gen RJMCMC", "wall_time"), 
    # ("gp/vi", "n_runs*L", "gp/sdvi", "SDVI", "inference_time"),
    # ("gp/smc", "n_particles", "gp/autogp", "AutoGP", "wall_time"),
    ]):
    
    df = df_from_json_dir(pathlib.Path(folder, model_path, "scale"), ["workload", "timings", "dcc_timings", "pconfig", "environment_info"], ["environ", "jax_environment"])
    # if model_path == "gmm":
    #     df = df[df["n_samples_per_chain"] == 2048]
    
    df["kind"] = df["gpu-brand"].map(lambda x: x[0][len("GPU 0: "):])
    df.loc[df["kind"] == "NVIDIA A100-SXM4-40GB", "kind"] = "NVIDIA A100S"
    df.loc[df["platform"] == "cpu", "kind"] = "CPU"
        
    if model_path == "gp/vi":
        df[SCALE_COL] = df["n_runs"] * df["L"]
        
    df = df[["platform", "kind", "num_workers", SCALE_COL, "total_time", "inference_time", "jax_total_jit_time", "n_available_devices"]]
    df = df.set_index(["platform", "kind", "num_workers", SCALE_COL], verify_integrity=True)
    df = df.sort_index()
    df = df.reset_index()
        
    df["METRIC"] = df["total_time"]
    METRIC_NAME = "wall time"
    
    row = {0: 0, 1: 0, 2: 3, 3: 3}[i]
    ax = axs#[row, i % 2]
    
    xticklabels = ["$2^{%d}$"%(n) for n in range(20+1)]
    ax.set_xticks(range(1,20+2), xticklabels, rotation=75)
    
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    yticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    # ax.tick_params(axis='y', which='major', labelsize=8, rotation=45, pad=0)
    # ax.tick_params(axis='y', which='minor', labelsize=8, rotation=45, pad=0)
    if i % 2 == 1:
        ax.yaxis.tick_right()
    ax.set_yticks(yticks, [f"{y:,d}s" for y in yticks])
    ax.set_ylim(7,2000)
    
    ax.grid(True)
    ax.set_axisbelow(True)
    for (platform, kind, n_devices), group in df.groupby(["platform", "kind", "num_workers"]):
        ax.plot(np.arange(len(group["METRIC"]))+1, group["METRIC"],
                #  label=f"{n_devices} {platform} {kind}", 
                c=get_color(kind, n_devices),
                marker=markers[(platform, n_devices)],
                linestyle=linestyles[kind],
                markersize=4,
                alpha=0.5
        )
    
    comp_df = df_from_json_dir(pathlib.Path(folder, comp_path), ["workload", "timings", "environment_info"], [])
    comp_df["METRIC"] = comp_df[comp_time]
    comp_df = comp_df.rename(columns={"cpu_count": "num_workers"})
    if model_path == "gp/vi":
        comp_df["n_runs"] = 1
        comp_df[SCALE_COL] = comp_df["n_runs"] * comp_df["L"]
        
    comp_df["kind"] = "COMP"
    comp_df = comp_df.set_index(["platform", "kind", "num_workers", SCALE_COL], verify_integrity=True)
    comp_df = comp_df.sort_index()
    comp_df = comp_df.reset_index()
    
    for (platform, kind, n_devices), group in comp_df.groupby(["platform", "kind", "num_workers"]):
        ax.plot(np.arange(len(group["METRIC"]))+1, group["METRIC"],
                # label=f"{n_devices} {platform} {kind}", 
                c=get_color(kind, n_devices),
                marker=markers[(platform, n_devices)],
                linestyle=linestyles["COMP"],
                markersize=4,
                alpha=0.5
        )
        
    
    legend_elements = device_legend_elements + [Line2D([0], [0], color=colors["COMP"], linestyle=linestyles["COMP"], lw=2, label=COMP_NAME)]
    fig.legend(handles=legend_elements, loc="center right")

# axs.set_title("Pedestrian: num.chains vs runtime")
# axs[0,1].set_title("b) Gaussian Mixture Model - RJMCMC")
# axs[3,0].set_title("c) Gaussian Process Model - VI")
# axs[3,1].set_title("d) Gaussian Process Model - SMC")

# axs.set_ylabel("Runtime [s]")
# axs[1,0].set_ylabel("$L_\\infty(\\hat{F},F)$ distance")
# axs[0,1].set_ylabel("Runtime [s]", rotation=270, labelpad=10)
# axs[0,1].yaxis.set_label_position("right")
# axs[1,1].set_ylabel("$L_\\infty(\\hat{F},F)$ distance", rotation=270, labelpad=16)
# axs[1,1].yaxis.set_label_position("right")

# axs[3,0].set_ylabel("Runtime [s]")
# axs[4,0].set_ylabel("SLP best ELBO")
# axs[3,1].set_ylabel("Runtime [s]", rotation=270, labelpad=10)
# axs[3,1].yaxis.set_label_position("right")
# axs[4,1].set_ylabel("SLP marginal likelihood", rotation=270, labelpad=16)
# axs[4,1].yaxis.set_label_position("right")



plt.tight_layout()

# fig.legend(handles=[legend_elements[i] for i in [0,4,1,5,2,6,3,7]], loc="upper center", ncols=4)
# fig.subplots_adjust(top=0.9)

# plt.savefig("scale_figure.png")
plt.savefig("scale_pedestrian.pdf")
plt.show()
        
        