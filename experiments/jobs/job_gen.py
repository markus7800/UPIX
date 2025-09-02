
CPU_TEMPLATE = """
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --partition=GPU-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=%d
#SBATCH --cpu-freq=high
#SBATCH --mail-user=markus.h.boeck@tuwien.ac.at
#SBATCH --mail-type=BEGIN,END,FAIL

export UV_PROJECT_ENVIRONMENT=.venv-cpu
uv sync --frozen --extra=cpu

%s

EOT
"""

CUDA_TEMPLATE = """
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --partition=GPU-l40s
#SBATCH --nodes=1
#SBATCH --nodelist=a-l40s-o-2
#SBATCH --gres=gpu:l40s:%d
#SBATCH --cpus-per-task=8
#SBATCH --cpu-freq=high
#SBATCH --mail-user=markus.h.boeck@tuwien.ac.at
#SBATCH --mail-type=BEGIN,END,FAIL

export UV_PROJECT_ENVIRONMENT=.venv-cuda
uv sync --frozen --extra=cuda

%s

EOT
"""

import subprocess

def sbatch(platform: str, jobname: str, ndevices: int, jobstr: str):
    assert platform in ("cpu", "cuda")
    template = CPU_TEMPLATE if platform == "cpu" else CUDA_TEMPLATE
    cmd = template % (jobname + f"_{ndevices}", ndevices, jobstr)
    print(cmd)
    subprocess.run(cmd, shell=True)
    

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("runner")
parser.add_argument("platform", help="cpu | cuda")
parser.add_argument("ndevices", type=int)
parser.add_argument("minpow", type=int)
parser.add_argument("maxpow", type=int)
args = parser.parse_args()

assert args.platform in ("cpu", "cuda")

pconfig_and_flags = {
    "experiments/runners/run_pedestrian_scale.py": "sequential pmap --no_progress",
    "experiments/runners/run_gp_vi_scale.py": "sequential smap_local --no_progress",
    "experiments/runners/run_gp_smc_scale.py": "sequential smap_local --no_progress",
    "experiments/runners/run_gmm_scale.py": "sequential pmap --no_progress",
}[args.runner]

jobname_prefix = {
    "experiments/runners/run_pedestrian_scale.py": "ped_",
    "experiments/runners/run_gp_vi_scale.py": "gp_vi_",
    "experiments/runners/run_gp_smc_scale.py": "gp_smc_",
    "experiments/runners/run_gmm_scale.py": "gp_smc_",
}[args.runner]
jobname = jobname_prefix + args.platform

# pows:
# ped:    00-20
# gmm:    ??-??
# gp vi:  00-13
# gm smc: 00-14

jobstr = f"python3 {args.runner} {args.platform} {args.ndevices} {args.minpow} {args.maxpow} {pconfig_and_flags}"

sbatch(args.platform, jobname, args.ndevices, jobstr)