import subprocess
import time
from scale_args import get_scale_args, MAX_TIME_S

platform, ndevices, minpow, maxpow, parallelisation, flags = get_scale_args()

n_slps = 8

N_ITER = 2048

NCHAINS = [2**n for n in range(minpow,maxpow+1)]
print(f"{NCHAINS=}")

RUNNER_T0 = time.monotonic()

if platform == "cpu":
    check_cmd = f"uv run --frozen -p python3.13 --extra=cpu experiments/runners/check_environ.py cpu {ndevices}"
    subprocess.run(check_cmd, shell=True, check=True)
    
    assert parallelisation == "sequential"
    vectorisation = "pmap"
    for nchains in NCHAINS:
        cmd = f"uv run --frozen -p python3.13 --extra=cpu evaluation/gmm/run_scale.py {parallelisation} {vectorisation} {n_slps} {nchains} {N_ITER} -host_device_count {ndevices} -num_workers {ndevices} --cpu {flags}"
        print('# CMD: ' + cmd)
        t0 = time.monotonic()
        subprocess.run(cmd, shell=True)
        elapsed = time.monotonic()-t0
        print(f"# Finished CMD in {elapsed:.3f}s")
        if elapsed > MAX_TIME_S:
            break
        
if platform == "cuda":
    check_cmd = f"uv run --frozen -p python3.13 --extra=cuda experiments/runners/check_environ.py gpu {ndevices}"
    subprocess.run(check_cmd, shell=True, check=True)
    
    assert parallelisation in ("sequential", "jax_devices")

    if parallelisation == "sequential":
        vectorisation = "smap_local"
        # smap_local uses less memory than pmap, but is slighlty slower
    else:
        vectorisation="vmap_global"
                
    for nchains in NCHAINS:
        cmd = f"uv run --frozen -p python3.13 --extra=cuda evaluation/gmm/run_scale.py {parallelisation} {vectorisation} {n_slps} {nchains} {N_ITER} -vmap_batch_size {2**19} -num_workers {ndevices} {flags}"
        print('# CMD: ' + cmd)
        t0 = time.monotonic()
        subprocess.run(cmd, shell=True)
        elapsed = time.monotonic()-t0
        print(f"# Finished CMD in {elapsed:.3f}s")
        if elapsed > MAX_TIME_S:
            break
            
print(f"\n# Runner finished in {time.monotonic() - RUNNER_T0:.3f}s")