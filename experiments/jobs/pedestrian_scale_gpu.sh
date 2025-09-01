#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=ped_gpu_$1
#SBATCH --partition=GPU-l40s
#SBATCH --nodes=1
#SBATCH --nodelist=a-l40s-o-2
#SBATCH --gres=gpu:l40s:$1
#SBATCH --cpus-per-task=8
#SBATCH --cpu-freq=high
#SBATCH --mail-user=markus.h.boeck@tuwien.ac.at
#SBATCH --mail-type=BEGIN,END,FAIL


export UV_PROJECT_ENVIRONMENT=.venv-cuda
uv sync --extra=cuda

python3 experiments/runners/run_pedestrian_scale.py cuda $1 20 sequential pmap --no_progress --no_colors

EOT
