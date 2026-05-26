#!/bin/bash

PROCESSES=$1
SEED=$2
ITERATIONS=${3:-1000000}

uv run -p python3.10 --no-project --with-requirements=evaluation/gp/sdvi/requirements.txt python3 evaluation/gp/sdvi/run_exp_pyro_extension.py \
    name=gp_grammar_sdvi \
    sdvi.exclusive_kl_num_particles=1 \
    sdvi.elbo_estimate_num_particles=100 \
    model=gp_kernel_learning \
    posterior_predictive_num_samples=1000 \
    sdvi.learning_rate=0.005 \
    sdvi.save_metrics_every_n=200 \
    resource_allocation=successive_halving \
    resource_allocation.num_total_iterations=$3 \
    sdvi.num_parallel_processes=$1 \
    sdvi.SCALE_EXPERIMENT=false \
    seed=$2