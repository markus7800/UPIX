import subprocess
import time
from scale_args import get_scale_args

platform, ndevices, minpow, maxpow, parallelisation, vectorisation, flags = get_scale_args()

n_slps = 8
n_iter = 1000
max_L = 8

Ks = [2**n for n in range(minpow,maxpow+1)]
print(f"{Ks=}")

RUNNER_T0 = time.monotonic()

if platform == "cpu":
    check_cmd = f"uv run --frozen -p python3.13 --extra=cpu experiments/runners/check_environ.py cpu {ndevices}"
    subprocess.run(check_cmd, shell=True, check=True)
    
    assert parallelisation == "sequential"
    assert vectorisation == "pmap"
    for K in Ks:
        if K // max_L == 0:
            n_runs = 1
            L = K
        else:
            n_runs = K // max_L
            L = max_L
        # have to set OMP_NUM_THREADS=1 otherwise crazy CPU over-util, do not really know why
        cmd = f"uv run --frozen -p python3.13 --extra=cpu --with-requirements=evaluation/gp/requirements.txt evaluation/gp/run_scale_vi.py {parallelisation} {vectorisation} {n_slps} {n_runs} {L} {n_iter} -host_device_count {ndevices} -num_workers {ndevices} -omp 1 --cpu {flags}"
        print('# CMD: ' + cmd)
        t0 = time.monotonic()
        subprocess.run(cmd, shell=True)
        print(f"# Finished CMD in {time.monotonic()-t0:.3f}s") 

if platform == "cuda":
    check_cmd = f"uv run --frozen -p python3.13 --extra=cuda experiments/runners/check_environ.py gpu {ndevices}"
    subprocess.run(check_cmd, shell=True, check=True)
    
    assert (parallelisation, vectorisation) in (("sequential", "pmap"), ("sequential", "vmap"),  ("jax_devices", "vmap"))
    if (parallelisation, vectorisation) == ("sequential", "vmap_local"):
        assert ndevices == 1 
    for K in Ks:
        if K // max_L == 0:
            n_runs = 1
            L = K
        else:
            n_runs = K // max_L
            L = max_L
        if n_runs == 1 and vectorisation == "vmap": vectorisation="vmap_local"
        if n_runs > 1 and vectorisation == "vmap": vectorisation="vmap_global"
        
        cmd = f"uv run --frozen -p python3.13 --extra=cuda --with-requirements=evaluation/gp/requirements.txt evaluation/gp/run_scale_vi.py {parallelisation} {vectorisation} {n_slps} {n_runs} {L} {n_iter} -vmap_batch_size {2**19} -num_workers {ndevices} {flags}"
        print('# ' + cmd)
        t0 = time.monotonic()
        subprocess.run(cmd, shell=True)
        print(f"# Finished CMD in {time.monotonic()-t0:.3f}s")

print(f"\n# Runner finished in {time.monotonic() - RUNNER_T0:.3f}s")