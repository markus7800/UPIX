import pathlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("folder")
args = parser.parse_args()
folder = args.folder

log_m2 = open(pathlib.Path(folder, "log_m2.txt"), "r").read()

log_cpu = open(pathlib.Path(folder, "log_cpu.txt"), "r").read()

log_l40s = open(pathlib.Path(folder, "log_l40s.txt"), "r").read()

log_a40 = open(pathlib.Path(folder, "log_a40.txt"), "r").read()

log_a100 = open(pathlib.Path(folder, "log_a100.txt"), "r").read()

import re
import matplotlib.pyplot as plt

time = [float(t) for t in re.findall(r"Finished inference task for N=\d+ in (\d+.\d+)s", log_cpu)]
l40s_time = [float(t) for t in re.findall(r"Finished inference task for N=\d+ in (\d+.\d+)s", log_l40s)]
a40_time = [float(t) for t in re.findall(r"Finished inference task for N=\d+ in (\d+.\d+)s", log_a40)]
a100_time = [float(t) for t in re.findall(r"Finished inference task for N=\d+ in (\d+.\d+)s", log_a100)]
factors_size = [int(f.replace(",","")) for f in re.findall(r"Factors size: ([\d,]+)", log_cpu)]
    
M = ["o", "^", "s", "D"]

fig, ax1 = plt.subplots(figsize=(5,2.5))

ax1.set_xlabel('number of balls in urn')
ax1.set_ylabel('time [s]')
ax1.set_yscale("log")
ax1.plot(time, marker=M[0], label="AMD EPYC 9355 256GB", zorder=3, alpha=0.5, color="tab:red")
# ax1.plot(a40_time, marker=M[2], label="A40 48GB", zorder=3, alpha=0.5)
ax1.plot(a100_time, marker=M[3], label="A100 80GB", zorder=3, alpha=0.5, color="tab:orange")
ax1.plot(l40s_time, marker=M[1], label="L40s 48GB", zorder=3, alpha=0.5, color="tab:blue")
leg = ax1.legend()
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_facecolor('none')
ax1.set_yticks([1,10], ["1", "10"], minor=False)
ax2 = ax1.twinx()

ax2.set_ylabel('sum of factor sizes', rotation=270, labelpad=14)
ax2.set_yscale("log")
ax2.plot(factors_size, marker=M[2], label="factor size", color="tab:gray", zorder=1, alpha=0.5)
ax2.tick_params(axis='y')
leg = ax2.legend(loc="lower right")
leg.get_frame().set_linewidth(0.0)
leg.get_frame().set_facecolor('none')
ax1.set_zorder(ax2.get_zorder()+1)
ax1.patch.set_visible(False)  # Make background transparent

fig.tight_layout()
plt.savefig("factor_size_scaling.pdf")
plt.show()