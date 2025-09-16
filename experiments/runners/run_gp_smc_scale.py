import subprocess
import time
from scale_args import get_scale_args

platform, ndevices, minpow, maxpow, parallelisation, flags = get_scale_args()

n_slps = 8

NPARTICLES = [2**n for n in range(minpow,maxpow+1)]
print(f"{NPARTICLES=}")

RUNNER_T0 = time.monotonic()

if platform == "cpu":
    check_cmd = f"uv run --frozen -p python3.13 --extra=cpu experiments/runners/check_environ.py cpu {ndevices}"
    subprocess.run(check_cmd, shell=True, check=True)
    
    assert parallelisation == "sequential"
    vectorisation = "smap_local"
    for nparticles in NPARTICLES:
        cmd = f"uv run --frozen -p python3.13 --extra=cpu --with-requirements=evaluation/gp/requirements.txt evaluation/gp/run_scale_smc.py {parallelisation} {vectorisation} {n_slps} {nparticles} -host_device_count {ndevices} -num_workers {ndevices} -omp 1 --cpu {flags}"
        print('# CMD: ' + cmd)
        t0 = time.monotonic()
        subprocess.run(cmd, shell=True)
        print(f"# Finished CMD in {time.monotonic()-t0:.3f}s")
        
if platform == "cuda":
    check_cmd = f"uv run --frozen -p python3.13 --extra=cuda experiments/runners/check_environ.py gpu {ndevices}"
    subprocess.run(check_cmd, shell=True, check=True)
    
    assert parallelisation in ("sequential", "jax_devices")

    if parallelisation == "sequential":
        vectorisation = "smap_local"
    else:
        vectorisation= "vmap_local"
        
    for nparticles in NPARTICLES:
        cmd = f"uv run --frozen -p python3.13 --extra=cuda --with-requirements=evaluation/gp/requirements.txt evaluation/gp/run_scale_smc.py {parallelisation} {vectorisation} {n_slps} {nparticles} -vmap_batch_size {2**19} -num_workers {ndevices} {flags}"
        print('# ' + cmd)
        t0 = time.monotonic()
        subprocess.run(cmd, shell=True)
        print(f"# Finished CMD in {time.monotonic()-t0:.3f}s")
        
print(f"\n# Runner finished in {time.monotonic() - RUNNER_T0:.3f}s")