
CPU_TEMPLATE = """
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={ncpus}
#SBATCH --cpu-freq=high
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.out

export UV_PROJECT_ENVIRONMENT=.venv-cpu
uv sync --frozen --extra=cpu

{jobstr}

EOT
"""

GRES = {
    "GPU-a40": "gpu:a40:",
    "GPU-l40s": "gpu:l40s:",
    "GPU-a100": "gpu:a100:",
    "GPU-a100s": "gpu:a100s:",
    "GPU-h100": "gpu:h100:"
}

CUDA_TEMPLATE = """
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition={partition}
#SBATCH --nodes=1{nodelist}
#SBATCH --gres={gres}
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.out

export UV_PROJECT_ENVIRONMENT=.venv-cuda-{jobname}
uv sync --frozen --extra=cuda

{jobstr}

rm -rf .venv-cuda-{jobname}

EOT
"""

import subprocess

def sbatch(platform: str, jobname_prefix: str, ndevices: int, jobstr: str, partition: str, node: str):
    assert platform in ("cpu", "cuda")
    if platform == "cpu":
        cmd = CPU_TEMPLATE.format(
            jobname=jobname_prefix + "_cpu_" + f"{ndevices:02d}_{partition[4:]}",
            partition=partition,
            ncpus=ndevices,
            jobstr=jobstr
        )
    else:
        if node != "":
            node_str = "\n#SBATCH --nodelist=%s" % node
        else:
            node_str = ""
        cmd = CUDA_TEMPLATE.format(
            jobname=jobname_prefix + "_cuda_" + f"{ndevices:1d}_{partition[4:]}",
            partition=partition,
            nodelist=node_str,
            gres=GRES[partition] + str(ndevices),
            jobstr=jobstr,
        )
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
    "experiments/runners/run_gmm_scale.py": "sequential pmap --no_progress",
}[args.runner]

jobname_prefix = {
    "experiments/runners/run_pedestrian_scale.py": "ped",
    "experiments/runners/run_gp_vi_scale.py": "gp_vi",
    "experiments/runners/run_gp_smc_scale.py": "gp_smc",
    "experiments/runners/run_gmm_scale.py": "gmm",
}[args.runner]

# pows:
# ped:    00-20
# gmm:    10-18 (19 for gpu? and 64 cpu?)
# gp vi:  00-13
# gp smc: 00-15

jobstr = f"python3 {args.runner} {args.platform} {args.ndevices} {args.minpow} {args.maxpow} {pconfig_and_flags}"

sbatch(args.platform, jobname_prefix, args.ndevices, jobstr, args.p, args.w)

# python3 experiments/jobs/job_gen.py experiments/runners/run_pedestrian_scale.py cuda 8 0 20
# python3 experiments/jobs/job_gen.py experiments/runners/run_gp_vi_scale.py cuda 8 0 13
# python3 experiments/jobs/job_gen.py experiments/runners/run_gp_smc_scale.py cuda 8 0 15
# python3 experiments/jobs/job_gen.py experiments/runners/run_gmm_scale.py cuda 8 10 18