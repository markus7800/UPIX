import numpy as np
import os
import pathlib
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("folder")
args = parser.parse_args()

datafolder = args.folder

results = [
    ("pedestrian/nonparametric/comp/cpu_08", "timings/inference_time", "result_metrics/L_inf"),
    ("pedestrian/comp/cpu_08", "timings/wall_time", "result_metrics/L_inf"),
    
    ("gp/sdvi/comp", "timings/wall_time", "result_metrics/lppd"),
    ("gp/vi/comp/cpu_08", "timings/wall_time", "result_metrics/lppd"),
    
    ("gmm/rjmcmc/comp/cpu_08", "timings/wall_time", "result/L_inf"),
    ("gmm/comp/cpu_08", "timings/wall_time", "result_metrics/L_inf"),
    
    ("gp/autogp/comp/cpu_08", "timings/wall_time", "result_metrics/lppd"),
    ("gp/smc/comp/cpu_08", "timings/wall_time", "result_metrics/lppd"),
    
    ("urn/dice/cpu_01", "timings/wall_time", "result_metrics/L_inf"),
    ("urn/ve/cpu_01", "timings/wall_time", "result_metrics/L_inf"),
    
    ("gp/sdvi/comp", "timings/wall_time", "result_metrics/lppd_std"),
    ("gp/vi/comp/cpu_08", "timings/wall_time", "result_metrics/lppd_std"),
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