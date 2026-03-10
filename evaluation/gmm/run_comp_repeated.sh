#!/bin/bash

for seed in {0..4} ;
do
    julia -t 8 --project=evaluation/gmm/gen evaluation/gmm/gen/gmm.jl 8 25000 $seed
done

for seed in {0..4} ;
do
    uv run -p python3.13 --frozen --extra=cpu evaluation/gmm/run_comp.py sequential pmap -host_device_count 8 -seed $seed
done
