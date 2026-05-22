#!/bin/bash

for seed in {0..4} ;
do
    bash evaluation/gp/sdvi run_comp.sh 10 $seed
done

for seed in {0..4} ;
do
    uv run -p python3.13 --frozen --extra=cpu --with pandas evaluation/gp/run_comp_vi.py cpu_multiprocess vmap_local -num_workers 10 -seed $seed
done
