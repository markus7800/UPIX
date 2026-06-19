import pandas as pd
import pathlib
import os
import numpy as np
import argparse
import json
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument("folder")

args = parser.parse_args()

folder = args.folder

ped_ground_truth = pathlib.Path(folder, "pedestrian", "groundtruth")

cdf_ests = []
for i in range(5):
    x = np.load(ped_ground_truth / f"gt_cdf-100-1_000_000_000_000-{i}.npy")
    cdf_ests.append(x)

cdf_ests = np.vstack(cdf_ests)
print("Pedestrian cdf std (averaged over grid) =", cdf_ests.std(axis=0).mean(), "+/-", cdf_ests.std(axis=0).std())


gmm_ground_truth = pathlib.Path(folder, "gmm", "groundtruth")

pdf_counts = []
for dirpath, _, files in os.walk(gmm_ground_truth):
    for file in files:
        if file.endswith(".json"):
            with open(pathlib.Path(dirpath, file), "r") as f:
                jdict = json.load(f)
                pdf_counts.append(np.array(jdict["result"]["counts"]))
                
max_len = max(len(counts) for counts in pdf_counts)
print([counts.sum() for counts in pdf_counts])
cdf_ests = np.vstack([np.cumsum(np.pad(counts, (0,max_len-len(counts)))) / counts.sum() for counts in pdf_counts])
print("GMM cdf std (averaged over K) =", cdf_ests.std(axis=0).mean(), "+/-", cdf_ests.std(axis=0).std())

