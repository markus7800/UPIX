#! /bin/bash

export TMPDIR=~/tmp
export OMP_NUM_THREADS=1


uv run --frozen -p python3.13 --extra=cuda evaluation/pedestrian/viz_mcmc_scale.py sequential vmap_global -vmap_batch_size 524288 -maxpow ${1:-20}

uv run --frozen -p python3.13 --extra=cuda --with=pandas evaluation/gp/viz_vi_elbo_scale.py sequential vmap_global -maxpow ${2:-14}

uv run --frozen -p python3.13 --extra=cuda --with=pandas evaluation/gp/viz_smc_particle_scale.py sequential vmap_local -maxpow ${3:-15}

uv run --frozen -p python3.13 --extra=cuda evaluation/gmm/viz_mcmc_scale.py sequential vmap_global -vmap_batch_size 524288 -maxpow ${4:-18}

# uv run --frozen -p python3.13 --extra=cpu --with=pandas evaluation/gp/viz_vi_elbo_scale_prelim.py sequential smap_local -host_device_count 8
# TODO: fix recompile at L=8 n_runs=8 for smap


# uv run --frozen -p python3.13 --extra=cpu evaluation/pedestrian/viz_mcmc_scale.py sequential pmap -vmap_batch_size 524288 -maxpow 20 -host_device_count 8

# uv run --frozen -p python3.13 --extra=cpu --with=pandas evaluation/gp/viz_vi_elbo_scale.py sequential smap_local -maxpow 14 -host_device_count 8

# uv run --frozen -p python3.13 --extra=cpu --with=pandas evaluation/gp/viz_smc_particle_scale.py sequential smap_local -maxpow 15 -host_device_count 8

# uv run --frozen -p python3.13 --extra=cpu evaluation/gmm/viz_mcmc_scale.py sequential pmap -vmap_batch_size 524288 -maxpow 18 -host_device_count 8
