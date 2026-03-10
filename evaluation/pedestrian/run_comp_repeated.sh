#!/bin/bash

for seed in {0..4} ;
do
    uv run -p python3.10 --no-project --with-requirements=evaluation/pedestrian/nonparametric-hmc/requirements.txt evaluation/pedestrian/nonparametric-hmc/pedestrian.py NP-DHMC 8 1000 100 -n_processes 8 --disable_bar -seed $seed
done

for seed in {0..4} ;
do
    uv run -p python3.13 --frozen --extra=cpu evaluation/pedestrian/run_comp.py sequential pmap -host_device_count 8 -seed $seed
done
