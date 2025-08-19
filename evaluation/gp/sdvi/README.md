Adapted from https://github.com/treigerm/sdvi_neurips

Python 3.10!!


```
python3 run_exp_pyro_extension.py \
    name=gp_grammar_sdvi \
    sdvi.exclusive_kl_num_particles=1 \
    sdvi.elbo_estimate_num_particles=100 \
    model=gp_kernel_learning \
    posterior_predictive_num_samples=10 \
    sdvi.learning_rate=0.005 \
    sdvi.save_metrics_every_n=200 \
    sdvi.num_parallel_processes=10 \
    resource_allocation=successive_halving \
    resource_allocation.num_total_iterations=1000000
```