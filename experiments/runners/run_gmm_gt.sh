#!/usr/bin/env bash
for seed in {0..4}; do
    julia -t 8 --project=evaluation/gmm/gen evaluation/gmm/gen/gmm.jl 8 3125000 $seed
done
