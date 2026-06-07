#! /bin/bash
# n_cpu_devices name

export TMPDIR=~/tmp
export OMP_NUM_THREADS=1

python3 experiments/runners/run_npdhmc_scale.py $1 0 ${3:-20} 2>&1 | tee ped_npdhmc_$1_$2.out

python3 experiments/runners/run_sdvi_scale.py $1 0 ${4:-14} 2>&1 | tee gp_sdvi_$1_$2.out

python3 experiments/runners/run_autogp_scale.py $1 0 ${5:-15} 2>&1 | tee gp_autogp_$1_$2.out

python3 experiments/runners/run_rjmcmc_scale.py $1 0 ${6:-18} 2>&1 | tee gmm_rjmcmc_$1_$2.out
