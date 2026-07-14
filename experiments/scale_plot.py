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
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("folder")

args = parser.parse_args()

folder = args.folder

def is_latex_available():
    required_cmds = ['latex', 'dvipng']
    return all(shutil.which(cmd) is not None for cmd in required_cmds)

plt.rcParams['text.usetex'] = is_latex_available()

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


gridspec = {
    "top": 0.9,
    "hspace": 0.0,
    "wspace": 0.05,
    "height_ratios": [1,1,0.8,1,1]
}
fig, axs = plt.subplots(5, 2, figsize=(8,11), gridspec_kw=gridspec)
axs[2,0].set_visible(False)
axs[2,1].set_visible(False)


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

has_series = {}

def get_marker(kind, platform, n_devices):
    return markers.get((platform, n_devices), M[0])

def get_linestyle(kind, n_devices):
    if kind not in linestyles:
        linestyles[kind] = "solid"
    return linestyles[kind]

available_colors = [
    "tab:gray"
    "tab:olive",
    "tab:cyan",
    "tab:purple",
]
def get_color(kind, n_devices):
    has_series[kind] = True
    if kind not in linestyles:
        colors[kind] = available_colors.pop()
    return colors[kind]

CPU_NAME = "2x AMD EPYC 9355"

for i, (model_path, SCALE_COL, comp_path, COMP_NAME, comp_time) in enumerate([
    ("pedestrian", "n_chains", "pedestrian/nonparametric", "NP-DHMC", "inference_time"),
    ("gmm", "n_chains", "gmm/rjmcmc", "Gen RJMCMC", "wall_time"), 
    ("gp/vi", "n_runs*L", "gp/sdvi", "SDVI", "inference_time"),
    ("gp/smc", "n_particles", "gp/autogp", "AutoGP", "wall_time"),
    ]):
    print(model_path, SCALE_COL, comp_path, COMP_NAME, comp_time)
    
    df = df_from_json_dir(pathlib.Path(folder, model_path, "scale"), ["workload", "timings", "dcc_timings", "pconfig", "environment_info"], ["environ", "jax_environment"])
    
    df["kind"] = df["gpu-brand"].map(lambda x: x[0][len("GPU 0: "):])
    df.loc[df["kind"] == "NVIDIA A100-SXM4-40GB", "kind"] = "NVIDIA A100S"
    df.loc[df["platform"] == "cpu", "kind"] = "CPU"
    
    if (df["platform"] == "cpu").any():
        if (df["cpu-brand"] != "AMD EPYC 9355 32-Core Processor").any():
            CPU_NAME = "CPU"
        
    if model_path == "gp/vi":
        df[SCALE_COL] = df["n_runs"] * df["L"]
        
    df = df[["platform", "kind", "num_workers", SCALE_COL, "total_time", "inference_time", "jax_total_jit_time", "n_available_devices"]]
    df = df.set_index(["platform", "kind", "num_workers", SCALE_COL])
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep='first')]
        print("Warning: index is not unique. Dropping duplicates.")
    assert df.index.is_unique
    df = df.sort_index()
    df = df.reset_index()
        
    df["METRIC"] = df["total_time"]
    METRIC_NAME = "wall time"
    
    row = {0: 0, 1: 0, 2: 3, 3: 3}[i]
    ax = axs[row, i % 2]
    
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
    # ax.tick_params(axis='y', which='major', labelsize=8, rotation=45, pad=0)
    # ax.tick_params(axis='y', which='minor', labelsize=8, rotation=45, pad=0)
    if i % 2 == 1:
        ax.yaxis.tick_right()
    ax.set_yticks(yticks, [f"{y:,d}" for y in yticks])
    ax.set_ylim(7,2000)
    # ax.set_xticklabels([])
    # hack to hide xticklabels because of shareaxis
    ax.tick_params(axis='x', which='major', rotation=90)
    ax.tick_params(axis='x', which='minor', rotation=90)
    
    ax.grid(True)
    ax.set_axisbelow(True)
    for (platform, kind, n_devices), group in df.groupby(["platform", "kind", "num_workers"]):
        ax.plot(np.arange(len(group["METRIC"]))+1, group["METRIC"],
                #  label=f"{n_devices} {platform} {kind}", 
                c=get_color(kind, n_devices),
                marker=get_marker(kind, platform, n_devices),
                linestyle=get_linestyle(kind, n_devices),
                markersize=4,
                alpha=0.5
        )
    
    comp_df = df_from_json_dir(pathlib.Path(folder, comp_path, "scale"), ["workload", "timings", "environment_info"], [])
    comp_df["METRIC"] = comp_df[comp_time]
    if comp_path != "gmm/rjmcmc":
        comp_df = comp_df.rename(columns={"cpu_count": "num_workers"})
    if model_path == "gp/vi":
        comp_df["n_runs"] = 1
        comp_df[SCALE_COL] = comp_df["n_runs"] * comp_df["L"]
        
    comp_df["kind"] = "COMP"
    comp_df = comp_df.set_index(["platform", "kind", "num_workers", SCALE_COL])
    if not comp_df.index.is_unique:
        comp_df = comp_df[~comp_df.index.duplicated(keep='first')]
        print("Warning: comp index is not unique. Dropping duplicates.")
    assert comp_df.index.is_unique
    comp_df = comp_df.sort_index()
    comp_df = comp_df.reset_index()
    
    for (platform, kind, n_devices), group in comp_df.groupby(["platform", "kind", "num_workers"]):
        ax.plot(np.arange(len(group["METRIC"]))+1, group["METRIC"],
                # label=f"{n_devices} {platform} {kind}", 
                c=get_color(kind, n_devices),
                marker=get_marker(kind, platform, n_devices),
                linestyle=linestyles["COMP"],
                markersize=4,
                alpha=0.5
        )
        
    
    legend_elements = [Line2D([0], [0], color=colors["COMP"], linestyle=linestyles["COMP"], lw=2, label=COMP_NAME)]
    ax.legend(handles=legend_elements, loc="upper left")

axs[0,0].set_title("a) Pedestrian Model - MCMC")
axs[0,1].set_title("b) Gaussian Mixture Model - RJMCMC")
axs[3,0].set_title("c) Gaussian Process Model - VI")
axs[3,1].set_title("d) Gaussian Process Model - SMC")

axs[0,0].set_ylabel("Runtime [s]")
axs[1,0].set_ylabel("$L_\\infty(\\hat{F},F)$ distance")
axs[0,1].set_ylabel("Runtime [s]", rotation=270, labelpad=10)
axs[0,1].yaxis.set_label_position("right")
axs[1,1].set_ylabel("$L_\\infty(\\hat{F},F)$ distance", rotation=270, labelpad=16)
axs[1,1].yaxis.set_label_position("right")

axs[3,0].set_ylabel("Runtime [s]")
axs[4,0].set_ylabel("SLP best ELBO")
axs[3,1].set_ylabel("Runtime [s]", rotation=270, labelpad=10)
axs[3,1].yaxis.set_label_position("right")
axs[4,1].set_ylabel("SLP marginal likelihood", rotation=270, labelpad=16)
axs[4,1].yaxis.set_label_position("right")



for (i, file) in enumerate([
    "pedestrian/scale/viz_ped_mcmc_scale_data.pkl",
    "gmm/scale/viz_gmm_mcmc_scale_data.pkl"
]):
    with open(pathlib.Path(folder, file), "rb") as f:
        res = pickle.load(f)
        n_chains_to_W1_distance, n_chains_to_infty_distance = res
        ax = axs[1,i]
        xticklabels = [f"{key:,}" for key in n_chains_to_infty_distance.keys()]
        ax.sharex(axs[0, i % 2])
        ax.boxplot(n_chains_to_infty_distance.values(), showfliers=False, patch_artist=True, boxprops=dict(facecolor="white", color="black")) # type: ignore
        ax.set_xticklabels(xticklabels, rotation=75)
        ax.set_xlabel("number of MCMC chains")
        if i % 2 == 1:
            ax.yaxis.tick_right()
        ax.grid(True)


for (i, (file,label,xticksformat)) in enumerate([
    ("gp/vi/scale/viz_gp_vi_elbo_scale_data.pkl", "log Z", "{:,} x {:,}"),
    ("gp/smc/scale/viz_gp_smc_particle_scale_data.pkl", "log Z", "{:,}")
]):
    with open(pathlib.Path(folder, file), "rb") as f:
        res = pickle.load(f)
        ax = axs[4,i]
        xticklabels = [xticksformat.format(*(key if isinstance(key, tuple) else [key])) for key in res.keys()]
        ax.sharex(axs[3, i % 2])
        ax.boxplot(res.values(), showfliers=False, patch_artist=True, boxprops=dict(facecolor="white", color="black")) # type: ignore
        ax.set_xticklabels(xticklabels, rotation=75)
        if i % 2 == 1:
            ax.yaxis.tick_right()
        ax.grid(True)
        
axs[4,0].set_xlabel("number of VI runs times number of samples per step")
axs[4,1].set_xlabel("number of SMC particles")


legend_elements = []
for kind, color in colors.items():
    if not has_series.get(kind, False):
        continue
    if kind == "COMP":
        label = "Reference on CPU"
    elif kind == "CPU":
        label = f"UPIX {CPU_NAME}"
    else:
        label = "UPIX " + kind
    legend_elements.append(Line2D([0], [0], color=color, linestyle=linestyles[kind], lw=2, label=label))
legend_elements.extend([Line2D([0],[0],alpha=0.)]*(4-len(legend_elements)))
ncols = len(legend_elements)

for i in range(4):
    n_cpu = 2**(i+3)
    n_gpu = 2**(i)
    legend_elements.append(Line2D([0],[0], marker=M[i], lw=0, color="black", label=f"{n_gpu} {"GPUs" if n_gpu > 1 else "GPU"} / {n_cpu} CPUs"))
legend_elements.extend([Line2D([0],[0],alpha=0.)]*(2*ncols-len(legend_elements)))

# plt.tight_layout()
half = len(legend_elements) // 2
reorder = [i // 2 if i % 2 == 0 else half + i // 2 for i in range(len(legend_elements))]

fig.legend(handles=[legend_elements[i] for i in reorder], loc="upper center", ncols=ncols)
# fig.subplots_adjust(top=0.9)

# plt.savefig("scale_figure.png")
os.makedirs("experiments/data/figures", exist_ok=True)
plt.savefig("experiments/data/figures/scale_figure.pdf")
plt.show()
        
        