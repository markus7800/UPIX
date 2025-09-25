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

fig, axs = plt.subplots(2, 4, sharex="col", figsize=(16,8))

for (i, (file, label)) in enumerate([
    ("viz_ped_mcmc_scale_data.pkl", "L_\\infty"),
    ("viz_gmm_mcmc_scale_data.pkl", "L_\\infty")
]):
    with open(pathlib.Path(folder, file), "rb") as f:
        res = pickle.load(f)

        n_chains_to_W1_distance, n_chains_to_infty_distance = res
        
        xticklabels = [f"{key:,}" for key in n_chains_to_infty_distance.keys()]
        axs[1,i].boxplot(n_chains_to_infty_distance.values(), showfliers=False) # type: ignore
        axs[1,i].set_xticklabels(xticklabels, rotation=75)
        axs[1,i].set_xlabel("number of MCMC chains")


for (i, (file,label,xticksformat)) in enumerate([
    ("viz_gp_vi_elbo_scale_data.pkl", "log Z", "{:,} x {:,}"),
    ("viz_gp_smc_particle_scale_data.pkl", "log Z", "{:,}")
]):
    with open(pathlib.Path(folder, file), "rb") as f:
        res = pickle.load(f)

        xticklabels = [xticksformat.format(*(key if isinstance(key, tuple) else [key])) for key in res.keys()]
        axs[1,i+2].boxplot(res.values(), showfliers=False) # type: ignore
        axs[1,i+2].set_xticklabels(xticklabels, rotation=75)
        
axs[1,2].set_xlabel("number of ADVI runs times number of samples per step")
axs[1,3].set_xlabel("number of SMC particles")

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

for i, (model_path, SCALE_COL, comp_path, COMP_NAME, comp_time) in enumerate([
    ("pedestrian", "n_chains", "pedestrian/nonparametric", "NPDHMC", "inference_time"),
    ("gmm", "n_chains", "gmm/rjmcmc", "Gen RJMCMC", "wall_time"), 
    ("gp/vi", "n_runs*L", "gp/sdvi", "SDVI", "inference_time"),
    ("gp/smc", "n_particles", "gp/autogp", "AutoGP", "wall_time"),
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
    
    ax = axs[0,i]
    
    # plt.xscale("log")
    # ax.xaxis.set_major_locator(LogLocator(base=2.0, subs=None))
    # ax.xaxis.set_minor_locator(LogLocator(base=2.0, subs=[]))
    # xticks = [2**n for n in range(20+1)]
    # ax.set_xticks(xticks, [f"{x:,d}" for x in xticks], rotation=75)
    # ax.set_xlabel(X_LABEL)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    yticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ax.set_yticks(yticks, [f"{y:,d}" for y in yticks])
    ax.set_ylim(15,2000)
    
    ax.grid(True)
    ax.set_axisbelow(True)
    for (platform, kind, n_devices), group in df.groupby(["platform", "kind", "num_workers"]):
        ax.plot(np.arange(len(group["METRIC"]))+1, group["METRIC"],
                #  label=f"{n_devices} {platform} {kind}", 
                c=get_color(kind, n_devices),
                marker=markers[(platform, n_devices)],
                linestyle=linestyles[kind],
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
                alpha=0.5
        )
        
    
    legend_elements = [Line2D([0], [0], color=colors["COMP"], linestyle=linestyles["COMP"], lw=2, label=COMP_NAME)]
    ax.legend(handles=legend_elements, loc="upper left")

axs[0,0].set_title("Pedestrian Model - MCMC")
axs[0,1].set_title("Gaussian Mixture Model - RJMCMC")
axs[0,2].set_title("Gaussian Process Model - VI")
axs[0,3].set_title("Gaussian Process Model - SMC")

axs[0,0].set_ylabel("Runtime [s]")
axs[1,0].set_ylabel("Approximation Quality")

axs[1,0].annotate("$L_\\infty(\\hat{F},F)$ distance to\nground truth posterior", (0.4, 0.4), fontsize=12, xycoords="axes fraction")
axs[1,1].annotate("$L_\\infty(\\hat{F},F)$ distance\nground truth posterior", (0.4, 0.4), fontsize=12, xycoords="axes fraction")
axs[1,2].annotate("local ELBO of SLP", (0.5, 0.4), fontsize=12, xycoords="axes fraction")
axs[1,3].annotate("local normalisation\nconstant of SLP", (0.5, 0.4), fontsize=12, xycoords="axes fraction")


legend_elements = []
for kind, color in colors.items():
    if kind == "COMP":
        label = "Reference"
    else:
        label = "UPIX " + kind
    legend_elements.append(Line2D([0], [0], color=color, linestyle=linestyles[kind], lw=2, label=label))
for i in range(4):
    n_cpu = 2**(i+3)
    n_gpu = 2**(i)
    legend_elements.append(Line2D([0],[0], marker=M[i], color="black", label=f"{n_gpu} {"GPUs" if n_cpu > 0 else "GPU"} / {n_cpu} CPUs"))
    
plt.tight_layout()

fig.legend(handles=legend_elements, loc="upper center", ncols=8)
fig.subplots_adjust(top=0.9)

plt.savefig("scale_figure.pdf")
plt.show()
        
        