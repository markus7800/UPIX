#!/usr/bin/env bash

export TMPDIR=$(pwd)/tmp

# ~25h total on 32 cpu
for seed in {0..4}; do
    julia -p 32 --project=evaluation/gmm/gen evaluation/gmm/gen/gmm.jl 32 3125000 $seed "groundtruth"
done