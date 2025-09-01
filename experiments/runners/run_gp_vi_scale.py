import subprocess
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("platform", help="cpu | cuda")
parser.add_argument("ndevices", type=int)
parser.add_argument("maxpow", type=int)
parser.add_argument("parallelisation")
parser.add_argument("vectorisaton")
parser.add_argument("--no_progress", action="store_true")
parser.add_argument("--no_colors", action="store_true")
args = parser.parse_args()

assert args.platform in ("cpu", "cuda")
ndevices = int(args.ndevices)
parallelisation = str(args.parallelisation)
vectorisaton = str(args.vectorisaton)
n_slps = 8
n_iter = 1000

Ls = [2**n for n in range(0,args.maxpow+1)]
print(f"{Ls=}")

flags = ""
if args.no_progress:
    flags += "--no_progress "
if args.no_save:
    flags += "--no_save "
    
if args.platform == "cpu":
    assert parallelisation == "sequential"
    assert vectorisaton == "smap_local"
    for L in Ls:
        # have to set OMP_NUM_THREADS=1 otherwise crazy CPU over-util, do not really know why
        cmd = f"uv run --frozen -p python3.13 --extra=cpu --with-requirements=evaluation/gp/requirements.txt evaluation/gp/run_scale_vi.py {parallelisation} {vectorisaton} {n_slps} {L} {n_iter} -host_device_count {ndevices} -num_workers {ndevices} -omp 1 --cpu {flags}"
        print('# CMD: ' + cmd) if args.no_colors else print('\033[95m' + cmd + '\033[0m')
        t0 = time.monotonic()
        subprocess.run(cmd, shell=True)
        print(f"# Finished CMD in {time.monotonic()-t0:.3f}s")
        

if args.platform == "cuda":
    assert (parallelisation, vectorisaton) in (("sequential", "smap_local"), ("sequential", "vmap_local"),  ("jax_devices", "vmap_local"))
    if (parallelisation, vectorisaton) == ("sequential", "vmap_local"):
        assert ndevices == 1 
    for L in Ls:
        cmd = f"uv run --frozen -p python3.13 --extra=cuda --with-requirements=evaluation/gp/requirements.txt evaluation/gp/run_scale_vi.py {parallelisation} {vectorisaton} {n_slps} {L} {n_iter} -vmap_batch_size {2**19} -num_workers {ndevices} {flags}"
        print('# ' + cmd) if args.no_colors else print('\033[95m' + cmd + '\033[0m')
        t0 = time.monotonic()
        subprocess.run(cmd, shell=True)
        print(f"# Finished CMD in {time.monotonic()-t0:.3f}s")