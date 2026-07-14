import numpy as np
import os
import pathlib
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("ncpu", type=int)
args = parser.parse_args()

datafolder = args.folder
ncpu = int(args.ncpu)

results = [
    (f"pedestrian/nonparametric/comp/cpu_{ncpu:02d}", "timings/inference_time", "result_metrics/L_inf"),
    (f"pedestrian/comp/cpu_{ncpu:02d}", "timings/wall_time", "result_metrics/L_inf"),
    
    (f"gp/sdvi/comp", "timings/wall_time", "result_metrics/lppd"),
    (f"gp/vi/comp/cpu_{ncpu:02d}", "timings/wall_time", "result_metrics/lppd"),
    
    (f"gmm/rjmcmc/comp/cpu_{ncpu:02d}", "timings/wall_time", "result/L_inf"),
    (f"gmm/comp/cpu_{ncpu:02d}", "timings/wall_time", "result_metrics/L_inf"),
    
    (f"gp/autogp/comp/cpu_{ncpu:02d}", "timings/wall_time", "result_metrics/lppd"),
    (f"gp/smc/comp/cpu_{ncpu:02d}", "timings/wall_time", "result_metrics/lppd"),
    
    ("urn/dice/cpu_01", "timings/wall_time", "result_metrics/L_inf"),
    ("urn/ve/cpu_01", "timings/wall_time", "result_metrics/L_inf"),
    
    # (f"gp/sdvi/comp", "timings/wall_time", "result_metrics/lppd_std"),
    # (f"gp/vi/comp/cpu_{ncpu:02d}", "timings/wall_time", "result_metrics/lppd_std"),
]

i = 0
for (folder, timepath, metricpath) in results:
    ts = []
    ms = []
    try:
        for file in os.listdir(pathlib.Path(datafolder, folder)):
            if not file.endswith(".json"): continue
            with open(pathlib.Path(datafolder, folder, file), "r") as f:
                j = json.load(f)
                t = j
                for k in timepath.split("/"): t = t[k]
                ts.append(t)
                m = j
                for k in metricpath.split("/"): m = m[k]
                ms.append(m)
        print(f"{folder}: {timepath}={np.mean(ts):.6g}±{np.std(ts):.6g} {metricpath}={np.mean(ms):.6g}±{np.std(ms):.6g}")
    except Exception as e:
        # raise e
        print("error")
    i += 1
    if i % 2 == 0: print()