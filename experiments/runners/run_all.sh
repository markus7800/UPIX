#! /bin/bash
# platform ndevices name

export UV_PROJECT_ENVIRONMENT=.venv-$1_$2_$3

uv sync --frozen --extra=$1

python3 experiments/runners/run_pedestrian_scale.py $1 $2 0 20 sequential pmap --no_progress 2>&1 | tee ped_$1_$2_$3.out

python3 experiments/runners/run_gp_vi_scale.py $1 $2 0 13 sequential smap_local --no_progress 2>&1 | tee gp_vi_$1_$2_$3.out

python3 experiments/runners/run_gp_smc_scale.py $1 $2 0 15 sequential smap_local --no_progress 2>&1 | tee gp_smc_$1_$2_$3.out

python3 experiments/runners/run_gmm_scale.py $1 $2 10 18 sequential pmap --no_progress 2>&1 | tee gmm_$1_$2_$3.out


rm -rf .venv-$1_$2_$3