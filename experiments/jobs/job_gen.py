
CPU_TEMPLATE = """
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=%s
#SBATCH --partition=%s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=%d
#SBATCH --cpu-freq=high
#SBATCH --output=R-%%x.%%j.out
#SBATCH --error=R-%%x.%%j.out
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
#SBATCH --partition=%s
#SBATCH --nodes=1
#SBATCH --nodelist=%s
#SBATCH --gres=gpu:l40s:%d
#SBATCH --cpus-per-task=8
#SBATCH --cpu-freq=high
#SBATCH --output=R-%%x.%%j.out
#SBATCH --error=R-%%x.%%j.out
#SBATCH --mail-user=markus.h.boeck@tuwien.ac.at
#SBATCH --mail-type=BEGIN,END,FAIL

export UV_PROJECT_ENVIRONMENT=.venv-cuda
uv sync --frozen --extra=cuda

%s

EOT
"""

import subprocess

def sbatch(platform: str, jobname: str, ndevices: int, jobstr: str, partition: str, node: str):
    assert platform in ("cpu", "cuda")
    if platform == "cpu":
        cmd = CPU_TEMPLATE % (jobname + f"_{ndevices:02d}", partition, ndevices, jobstr)
    else:
        cmd = CUDA_TEMPLATE % (jobname + f"_{ndevices:1d}", partition, node, ndevices, jobstr)
    print(cmd)
    subprocess.run(cmd, shell=True)
    

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("runner")
parser.add_argument("platform", help="cpu | cuda")
parser.add_argument("ndevices", type=int)
parser.add_argument("minpow", type=int)
parser.add_argument("maxpow", type=int)
parser.add_argument("-p", type=str, default="GPU-l40s")
parser.add_argument("-w", type=str, default="a-l40s-o-2")
args = parser.parse_args()

assert args.platform in ("cpu", "cuda")

pconfig_and_flags = {
    "experiments/runners/run_pedestrian_scale.py": "sequential pmap --no_progress",
    "experiments/runners/run_gp_vi_scale.py": "sequential smap_local --no_progress",
    "experiments/runners/run_gp_smc_scale.py": "sequential smap_local --no_progress",
    "experiments/runners/run_gmm_scale.py": "sequential smap_local --no_progress",
}[args.runner]

jobname_prefix = {
    "experiments/runners/run_pedestrian_scale.py": "ped_",
    "experiments/runners/run_gp_vi_scale.py": "gp_vi_",
    "experiments/runners/run_gp_smc_scale.py": "gp_smc_",
    "experiments/runners/run_gmm_scale.py": "gmm_",
}[args.runner]
jobname = jobname_prefix + args.platform

# pows:
# ped:    00-20
# gmm:    10-18 (19 for gpu? and 64 cpu?)
# gp vi:  00-13
# gp smc: 00-15

jobstr = f"python3 {args.runner} {args.platform} {args.ndevices} {args.minpow} {args.maxpow} {pconfig_and_flags}"

sbatch(args.platform, jobname, args.ndevices, jobstr, args.p, args.w)

# python3 experiments/jobs/job_gen.py experiments/runners/run_pedestrian_scale.py cuda 8 0 20
# python3 experiments/jobs/job_gen.py experiments/runners/run_gp_vi_scale.py cuda 8 0 13
# python3 experiments/jobs/job_gen.py experiments/runners/run_gp_smc_scale.py cuda 8 0 15
# python3 experiments/jobs/job_gen.py experiments/runners/run_gmm_scale.py cuda 8 10 18