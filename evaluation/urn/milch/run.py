
import subprocess
import re
import numpy as np
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("-seed", default=0, type=int, required=False)
parser.add_argument("--show_plots", action="store_true")
args = parser.parse_args()

res = subprocess.run(["./urn_biased.out", str(args.seed * 10000)], capture_output=True)

out = res.stdout.decode()
print(out)
print(res.stderr.decode())

match = re.findall(r"(\d+) -> (\d.\d+e-\d\d)", out)

gt = np.load("../gt_ps.npy")
urn_result = np.zeros(len(gt))
print(gt)

for ix, p in match:
    urn_result[int(ix)-1] = float(p)


err = np.abs(urn_result - gt)
l_inf = np.max(err)
print("Max err:", l_inf)


match = re.findall(r"running time: (\d+.\d+s)", out)
timing = str(match[0])
print("Timings:", timing)


if args.show_plots:
    import matplotlib.pyplot as plt
    plt.plot(err)
    plt.show()
    
import pathlib, json, uuid
from datetime import datetime
import cpuinfo

def get_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return int(len(os.sched_getaffinity(0))) # type: ignore
    else:
        return int(os.cpu_count()) # type: ignore
def _get_last_git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'log',  '--format=%H', '-n', '1']).decode().rstrip()
    except:
        return ""

platform = "cpu"
num_workers = 1
id_str = str(uuid.uuid4())
json_result = {
    "id": id_str,
    "workload": {
        "seed": args.seed,
        "n_iter": 10000000,
    },
    "result_metrics": {
        "L_inf": l_inf
    },
    "timings": {
        "inference_time": timing
    },
    "environment_info": {
        "platform": "cpu",
        "cpu-brand": cpuinfo.get_cpu_info()["brand_raw"],
        "cpu_count": get_cpu_count(),
        "git_commit": _get_last_git_commit(),
    }
}
now = datetime.today().strftime('%Y-%m-%d_%H-%M')
fpath = pathlib.Path(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..",
    "experiments", "data", "urn", "swift", f"{platform}_{num_workers:02d}",
    f"date_{now}_{id_str[:8]}.json")
fpath.parent.mkdir(exist_ok=True, parents=True)
with open(fpath, "w") as f:
    json.dump(json_result, f, indent=2)