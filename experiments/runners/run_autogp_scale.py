

import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ndevices", type=int)
parser.add_argument("minpow", type=int)
parser.add_argument("maxpow", type=int)
args = parser.parse_args()

NPARTICLES = [2**n for n in range(args.minpow,args.maxpow+1)]

CMD_TEMPLATE = "OMP_NUM_THREADS=1 julia -t %d --project=evaluation/gp/autogp evaluation/gp/autogp/main.jl %d false"

RUNNER_T0 = time.monotonic()

for nparticles in NPARTICLES:
    cmd = CMD_TEMPLATE % (args.ndevices, nparticles)
    print('# CMD: ' + cmd)
    t0 = time.monotonic()
    subprocess.run(cmd, shell=True)
    print(f"# Finished CMD in {time.monotonic()-t0:.3f}s")
        
print(f"\n# Runner finished in {time.monotonic() - RUNNER_T0:.3f}s")