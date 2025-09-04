import subprocess
import time
from scale_args import get_scale_args

platform, ndevices, minpow, maxpow, parallelisation, vectorisation, flags = get_scale_args()

n_slps = 8

N_ITERS = [256, 2048]

NCHAINS = [2**n for n in range(minpow,maxpow+1)]
print(f"{NCHAINS=}")

RUNNER_T0 = time.monotonic()

if platform == "cpu":
    check_cmd = f"uv run --frozen -p python3.13 --extra=cpu experiments/runners/check_environ.py cpu {ndevices}"
    subprocess.run(check_cmd, shell=True, check=True)
    
    assert (parallelisation, vectorisation) in (("sequential", "pmap"), ("sequential", "smap_local"))
    for n_iter in N_ITERS:
        for nchains in NCHAINS:
            cmd = f"uv run --frozen -p python3.13 --extra=cpu evaluation/gmm/run_scale.py {parallelisation} {vectorisation} {n_slps} {nchains} {n_iter} -host_device_count {ndevices} -num_workers {ndevices} --cpu {flags}"
            print('# CMD: ' + cmd)
            t0 = time.monotonic()
            subprocess.run(cmd, shell=True)
            print(f"# Finished CMD in {time.monotonic()-t0:.3f}s")
        
if platform == "cuda":
    check_cmd = f"uv run --frozen -p python3.13 --extra=cuda experiments/runners/check_environ.py gpu {ndevices}"
    subprocess.run(check_cmd, shell=True, check=True)
    
    # smap_local uses less memory than pmap, but is slighlty slower
    assert (parallelisation, vectorisation) in (("sequential", "pmap"), ("sequential", "smap_local"), ("sequential", "vmap_global"),  ("jax_devices", "vmap_global"))
    if (parallelisation, vectorisation) == ("sequential", "vmap_global"):
        assert ndevices == 1 
    for n_iter in N_ITERS:
        for nchains in NCHAINS:
            cmd = f"uv run --frozen -p python3.13 --extra=cuda evaluation/gmm/run_scale.py {parallelisation} {vectorisation} {n_slps} {nchains} {n_iter} -vmap_batch_size {2**19} -num_workers {ndevices} {flags}"
            print('# CMD: ' + cmd)
            t0 = time.monotonic()
            subprocess.run(cmd, shell=True)
            print(f"# Finished CMD in {time.monotonic()-t0:.3f}s")
            
print(f"\n# Runner finished in {time.monotonic() - RUNNER_T0:.3f}s")