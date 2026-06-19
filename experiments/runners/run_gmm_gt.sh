#!/usr/bin/env bash

# ~50h total on 8 cpu
for seed in {0..4}; do
    julia -p 8 --project=evaluation/gmm/gen evaluation/gmm/gen/gmm.jl 8 3125000 $seed
done
