import argparse
import re
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()


with open(args.filename, "r") as f:
    s = f.read()
    time = []
    for m in re.findall(r"^real\t(\d+)m([\d.]+)s", s, flags=re.MULTILINE):
        time.append(int(m[0])*60 + float(m[1]))
    time = np.array(time)
    print(np.mean(time), np.std(time))
    
    gt = np.load("evaluation/urn/gt_ps.npy")
    urn_result = np.zeros(len(gt))
    
    for r in re.findall(r"^(\d+)\t([\d.e-]+)", s, flags=re.MULTILINE):
        urn_result[int(r[0])-1] = float(r[1])
        

    err = np.abs(urn_result - gt)
    l_inf = np.max(err)
    print("Max err:", l_inf)
