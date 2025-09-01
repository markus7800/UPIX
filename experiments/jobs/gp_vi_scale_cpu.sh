#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=gp_vi_cpu
#SBATCH --partition=GPU-l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=$1
#SBATCH --cpu-freq=high

python3 experiments/runners/run_gp_vi_scale.py cpu $1 20 sequential pmap --no_progress

EOT