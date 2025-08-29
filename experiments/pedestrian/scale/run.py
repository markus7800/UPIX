import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("platform", help="cpu | cuda")
parser.add_argument("ndevices", type=int)
parser.add_argument("maxpow", type=int)
parser.add_argument("parallelisation")
parser.add_argument("vectorisaton")
args = parser.parse_args()

assert args.platform in ("cpu", "cuda")
ndevices = int(args.ndevices)
parallelisation = str(args.parallelisation)
vectorisaton = str(args.vectorisaton)

NCHAINS = [2**n for n in [3, 10, 13, 16, 17, 18, 19, 20] if n <= args.maxpow]
print(NCHAINS)


if args.platform == "cpu":
    assert parallelisation == "sequential"
    assert vectorisaton == "pmap"
    for nchains in NCHAINS:
        cmd = f"uv run --frozen -p python3.13 --extra=cpu evaluation/pedestrian/run_scale.py {parallelisation} {vectorisaton} 8 {nchains} 256 -host_device_count {ndevices} -num_workers {ndevices}"
        print('\033[95m' + cmd + '\033[0m')
        subprocess.run(cmd, shell=True)
        

if args.platform == "cuda":
    assert (parallelisation, vectorisaton) in (("sequential", "pmap"), ("sequential", "vmap_global"),  ("jax_devices", "vmap_global"))
    if (parallelisation, vectorisaton) == ("sequential", "vmap_global"):
        assert ndevices == 1 
    for nchains in NCHAINS:
        cmd = f"uv run --frozen -p python3.13 --extra=cuda evaluation/pedestrian/run_scale.py {parallelisation} {vectorisaton} 8 {nchains} 256 -vmap_batch_size {2**19} -num_workers {ndevices}"
        print('\033[95m' + cmd + '\033[0m')
        subprocess.run(cmd, shell=True)
        
        
# python3 experiments/pedestrian/scale/run.py cuda 8 20 sequential pmap
