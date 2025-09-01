#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=gp_vi_cpu_$1
#SBATCH --partition=GPU-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=$1
#SBATCH --cpu-freq=high
#SBATCH --mail-user=markus.h.boeck@tuwien.ac.at
#SBATCH --mail-type=BEGIN,END,FAIL

export UV_PROJECT_ENVIRONMENT=.venv-cpu
uv sync --extra=cpu

python3 experiments/runners/run_gp_vi_scale.py cpu $1 12 sequential smap_local --no_progress --no_colors

EOT