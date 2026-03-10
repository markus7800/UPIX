#!/bin/bash

for seed in {0..4} ;
do
    uv run --with numpy,matplotlib --directory evaluation/urn/milch run.py -seed $seed
done

for seed in {0..4} ;
do
    uv run -p python3.13 --frozen --extra=cpu evaluation/urn/run_comp.py sequential vmap_local 20 --jit_inf -seed $seed
done
