
import subprocess
import re
import numpy as np

res = subprocess.run(["./urn_biased.out"], capture_output=True)

out = res.stdout.decode()
print(out)

match = re.findall(r"(\d+) -> (\d.\d+e-\d\d)", out)

urn_result = [float(p) for ix, p in match]
assert (np.array([int(ix) for ix, p in match]) == np.arange(1,len(urn_result)+1)).all()

urn_result = np.array(urn_result)

gt = np.load("../gt_ps.npy")

err = np.abs(np.hstack((urn_result,np.zeros(len(gt)-len(urn_result),))) - gt)

print("Max err:", np.max(err))

import matplotlib.pyplot as plt
plt.plot(err)
plt.show()

