

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

for nchains in NCHAINS:
    cmd = CMD_TEMPLATE % (args.ndevices, nchains, 256)
    print('# CMD: ' + cmd)
    t0 = time.monotonic()
    subprocess.run(cmd, shell=True)
    print(f"# Finished CMD in {time.monotonic()-t0:.3f}s")
        