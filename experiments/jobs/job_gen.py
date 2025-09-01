
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
uv sync --extra=cpu

%s

EOT
"""

GPU_TEMPLATE = """
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
uv sync --extra=cuda

%s

EOT
"""

import subprocess

def sbatch(platform: str, jobname: str, ndevices: int, jobstr: str):
    assert platform in ("CPU", "GPU")
    template = CPU_TEMPLATE if platform == "CPU" else GPU_TEMPLATE
    cmd = template % (jobname + f"_{ndevices}", ndevices, jobstr)
    print(cmd)
    subprocess.run(cmd, shell=True)
    
