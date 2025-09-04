

import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ndevices", type=int)
parser.add_argument("minpow", type=int)
parser.add_argument("maxpow", type=int)
args = parser.parse_args()

NCHAINS = [2**n for n in range(args.minpow,args.maxpow+1)]

CMD_TEMPLATE = "julia -t %d --project=evaluation/gmm/gen evaluation/gmm/gen/gmm.jl %d %d"

RUNNER_T0 = time.monotonic()

for n_iter in [256, 2048]:
    for nchains in NCHAINS:
        cmd = CMD_TEMPLATE % (args.ndevices, nchains, n_iter)
        print('# CMD: ' + cmd)
        t0 = time.monotonic()
        subprocess.run(cmd, shell=True)
        print(f"# Finished CMD in {time.monotonic()-t0:.3f}s")
        
print(f"\n# Runner finished in {time.monotonic() - RUNNER_T0:.3f}s")