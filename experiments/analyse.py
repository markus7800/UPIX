import pandas as pd
import pathlib
import os
import json
import matplotlib.pyplot as plt

path = pathlib.Path("experiments", "pedestrian", "scale")

results = []
for file in os.listdir(path):
    if file.endswith(".json"):
        with open(path.joinpath(file), "r") as f:
            jdict = json.load(f)
            print(list(jdict.keys()))
            df_dict = jdict["workload"] | jdict["timings"] | jdict["dcc_timings"] | jdict["pconfig"] | jdict["environment_info"]
            del df_dict["environ"]
            del df_dict["jax_environment"]
            results.append(df_dict)
            
df = pd.DataFrame(results)
print(df)
plt.scatter(df["n_chains"], df["inference_time"])
plt.xscale("log")
plt.show()