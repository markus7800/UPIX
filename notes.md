JAX_LOG_COMPILES=1

## slurm

srun -p GPU-l40s -w a-l40s-o-1 --gres=gpu:l40s:8 -c 64 --cpu-freq=high --pty bash

## Pedestrian

### Ground truth
~ 1h
uv run --frozen -p python3.13 --extra=cuda evaluation/pedestrian/ground_truth.py sequential vmap_global 100 1_000_000 1000 1_000 

~ 4h
uv run --frozen -p python3.13 --extra=cuda evaluation/pedestrian/ground_truth.py sequential vmap_global 1_000 1_000_000 1000 1_000 

### Scale

uv run --frozen -p python3.13 --extra=cuda evaluation/pedestrian/run_scale.py sequential pmap 8 1048576 256 -num_workers ...

uv run --frozen --python=python3.13 --extra=cuda evaluation/pedestrian/run_scale.py jax_devices vmap_glbal 8 1048576 256 -vmap_batch_size 524288 -num_workers ...

uv run --frozen --python=python3.13 --extra=cpu evaluation/pedestrian/run_scale.py sequential pmap 8 1048576 256 --cpu -host_device_count ...

### Nonparametric HMC

uv run -p python3.10 --no-project --with-requirements=evaluation/pedestrian/nonparametric-hmc/requirements.txt evaluation/pedestrian/nonparametric-hmc/pedestrian.py NP-DHMC 8 256 0 -n_processes 8  --disable_bar


## GP

uv run  --frozen --python=python3.13 --extra=cpu --with=pandas evaluation/gp/run_comp_vi.py cpu_multiprocess vmap_local --cpu -omp 1 -num_workers=...

uv run  --frozen --python=python3.13 --extra=cuda --with=pandas evaluation/gp/run_scale_vi.py sequential smap_local 1 1 1000 -host_device_count 64 --cpu



# example section

cd evaluation/pedestrian/nonparametric-hmc
uv run -p python3.10 --no-project --with-requirements=requirements.txt pedestrian.py NP-DHMC 8 1000 100 -n_processes 8  --store_samples
uv run -p python3.10 --no-project --with-requirements=requirements.txt check_results.py

uv run evaluation/pedestrian/run_comp.py sequential pmap --show_plots -host_device_count 8

bash evaluation/gp/sdvi/run_comp.sh 10
uv run -p python3.13 --with pandas evaluation/gp/run_comp_vi.py cpu_multiprocess vmap_local

uv run --with=pandas evaluation/gp/run_comp_smc.py sequential smap_local -host_device_count 10 --show_plots

uv run  evaluation/urn/run_comp.py sequential vmap_local 20 --show_plots


uv run evaluation/gmm/run_comp.py sequential pmap -host_device_count 8

julia -t 8 --project=evaluation/gmm/gen evaluation/gmm/gen/gmm.jl 8 25000