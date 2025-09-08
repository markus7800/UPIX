

import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ndevices", type=int)
parser.add_argument("minpow", type=int)
parser.add_argument("maxpow", type=int)
args = parser.parse_args()

Ls = [2**n for n in range(args.minpow,args.maxpow+1)]

CMD_TEMPLATE = """
uv run -p python3.10 --no-project --with-requirements=evaluation/gp/sdvi/requirements.txt python3 evaluation/gp/sdvi/run_exp_pyro_extension.py \
    name=gp_grammar_sdvi \
    sdvi.elbo_estimate_num_particles=100 \
    model=gp_kernel_learning \
    posterior_predictive_num_samples=10 \
    sdvi.learning_rate=0.005 \
    sdvi.save_metrics_every_n=200 \
    resource_allocation=successive_halving \
    resource_allocation.num_total_iterations=1000000 \
    sdvi.num_parallel_processes=%d \
    sdvi.exclusive_kl_num_particles=%d \
    sdvi.SCALE_EXPERIMENT=true
"""

RUNNER_T0 = time.monotonic()

for L in Ls:
    cmd = CMD_TEMPLATE % (args.ndevices, L)
    print('# CMD: ' + cmd)
    t0 = time.monotonic()
    subprocess.run(cmd, shell=True)
    print(f"# Finished CMD in {time.monotonic()-t0:.3f}s")
            
print(f"\n# Runner finished in {time.monotonic() - RUNNER_T0:.3f}s")