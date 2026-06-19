#! /bin/bash

export TMPDIR=~/tmp
export OMP_NUM_THREADS=1

# l40s 150min
uv run --frozen -p python3.13 --extra=cuda evaluation/pedestrian/viz_mcmc_scale.py sequential vmap_global -vmap_batch_size 524288 -maxpow ${1:-20} --no_progress

# l40s 37min
uv run --frozen -p python3.13 --extra=cuda --with=pandas evaluation/gp/viz_vi_elbo_scale.py sequential vmap_global -maxpow ${2:-14} --no_progress

# l40s ~20min
uv run --frozen -p python3.13 --extra=cuda --with=pandas evaluation/gp/viz_smc_particle_scale.py sequential vmap_local -maxpow ${3:-15} --no_progress

# l40s 392min, but wrong results
# uv run --frozen -p python3.13 --extra=cuda evaluation/gmm/viz_mcmc_scale.py sequential vmap_global -vmap_batch_size 524288 -maxpow ${4:-18} --no_progress

# uv run --frozen -p python3.13 --extra=cpu --with=pandas evaluation/gp/viz_vi_elbo_scale_prelim.py sequential smap_local -host_device_count 8 --no_progress

# uv run --frozen -p python3.13 --extra=cpu evaluation/pedestrian/viz_mcmc_scale.py sequential pmap -vmap_batch_size 524288 -maxpow 20 -host_device_count 8 --no_progress

# uv run --frozen -p python3.13 --extra=cpu --with=pandas evaluation/gp/viz_vi_elbo_scale.py sequential smap_local -maxpow 14 -host_device_count 8 --no_progress

# uv run --frozen -p python3.13 --extra=cpu --with=pandas evaluation/gp/viz_smc_particle_scale.py sequential smap_local -maxpow 15 -host_device_count 8 --no_progress

uv run --frozen -p python3.13 --extra=cpu evaluation/gmm/viz_mcmc_scale.py sequential pmap -vmap_batch_size 524288 -maxpow 18 -host_device_count 8 --no_progress
