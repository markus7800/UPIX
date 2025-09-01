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
n_iter = 256

NCHAINS = [2**n for n in range(3,args.maxpow+1)]
print(f"{NCHAINS=}")

progress = "--no_progress" if args.no_progress else ""

if args.platform == "cpu":
    assert parallelisation == "sequential"
    assert vectorisaton == "pmap"
    for nchains in NCHAINS:
        cmd = f"uv run --frozen -p python3.13 --extra=cpu evaluation/pedestrian/run_scale.py {parallelisation} {vectorisaton} {n_slps} {nchains} {n_iter} -host_device_count {ndevices} -num_workers {ndevices} --cpu {progress}"
        print('## ' + cmd) if args.no_colors else print('\033[95m' + cmd + '\033[0m')
        t0 = time.monotonic()
        subprocess.run(cmd, shell=True)
        print(f"Finished in {time.monotonic()-t0:.3f}s")
        

if args.platform == "cuda":
    assert (parallelisation, vectorisaton) in (("sequential", "pmap"), ("sequential", "vmap_global"),  ("jax_devices", "vmap_global"))
    if (parallelisation, vectorisaton) == ("sequential", "vmap_global"):
        assert ndevices == 1 
    for nchains in NCHAINS:
        cmd = f"uv run --frozen -p python3.13 --extra=cuda evaluation/pedestrian/run_scale.py {parallelisation} {vectorisaton} {n_slps} {nchains} {n_iter} -vmap_batch_size {2**19} -num_workers {ndevices} {progress}"
        print('## ' + cmd) if args.no_colors else print('\033[95m' + cmd + '\033[0m')
        t0 = time.monotonic()
        subprocess.run(cmd, shell=True)
        print(f"Finished in {time.monotonic()-t0:.3f}s")