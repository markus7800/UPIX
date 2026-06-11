#!/usr/bin/env bash
for seed in {0..4}; do
    uv run --python python3.13 --extra=cuda --frozen evaluation/pedestrian/ground_truth.py sequential vmap_global 100 1_000_000 1_000 1_000 $seed
done
