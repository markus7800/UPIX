

import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ndevices", type=int)
parser.add_argument("minpow", type=int)
parser.add_argument("maxpow", type=int)
args = parser.parse_args()

NCHAINS = [2**n for n in range(args.minpow,args.maxpow+1)]

CMD_TEMPLATE = """
uv run -p python3.10 --no-project --with-requirements=evaluation/pedestrian/nonparametric-hmc/requirements.txt evaluation/pedestrian/nonparametric-hmc/pedestrian.py NP-DHMC %d 256 0 -n_processes %d --disable_bar
"""

RUNNER_T0 = time.monotonic()

for nchains in NCHAINS:
    cmd = CMD_TEMPLATE % (nchains, args.ndevices)
    print('# CMD: ' + cmd)
    t0 = time.monotonic()
    subprocess.run(cmd, shell=True)
    elapsed = time.monotonic()-t0
    print(f"# Finished CMD in {elapsed:.3f}s")
    if elapsed > 5000:
        break
        
print(f"\n# Runner finished in {time.monotonic() - RUNNER_T0:.3f}s")