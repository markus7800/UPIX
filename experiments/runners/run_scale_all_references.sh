#! /bin/bash
# ndevices name

python3 experiments/runners/run_npdhmc_scale.py $1 0 20 2>&1 | tee ped_npdhmc_$1_$2.out

python3 experiments/runners/run_sdvi_scale.py $1 0 14 2>&1 | tee gp_sdvi_$1_$2.out

python3 experiments/runners/run_autogp_scale.py $1 0 15 2>&1 | tee gp_autogp_$1_$2.out

python3 experiments/runners/run_rjmcmc_scale.py $1 0 18 2>&1 | tee gmm_rjmcmc_$1_$2.out
