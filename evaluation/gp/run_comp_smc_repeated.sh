#!/bin/bash

for seed in {0..4} ;
do
    julia -t 10 --project=evaluation/gp/autogp evaluation/gp/autogp/main.jl 100 false $seed ;
done

for seed in {0..4} ;
do
    uv run -p python3.13 --frozen --extra=cpu --with=pandas evaluation/gp/run_comp_smc.py sequential smap_local -host_device_count 10 -seed $seed --show_plots ;
done
