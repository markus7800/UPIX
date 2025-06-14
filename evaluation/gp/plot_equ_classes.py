import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation/gp/tree_counts.csv", delimiter=" ")
print(df)
# print(df.dtypes)

c1 = np.mean(np.diff(np.log(df["n_trees_f"])))
c2 = np.mean(np.diff(np.log(df["n_equ_trees_f"])))
print(np.exp(c1))
print(np.exp(c2))

fig, ax1 = plt.subplots()
ax1.set_xlabel('number of leaves')
ax1.set_ylabel('number of trees')
ax1.plot(df["n_leaves"], df["n_trees_f"])
ax1.plot(df["n_leaves"], df["n_equ_trees_f"])
# ax1.plot(df["n_leaves"], np.exp(c1 * (df["n_leaves"]-1)), color="tab:blue", linestyle="--")
# ax1.plot(df["n_leaves"], np.exp(c2 * (df["n_leaves"]-1)), color="tab:orange", linestyle="--")
ax1.set_yscale("log")

ax2 = ax1.twinx()
ax2.set_ylabel('probability')
# ax2.plot(df["p"], color="tab:red")
ax2.plot(df["n_leaves"], df["p_tree"], color="tab:green")
ax2.set_yscale("log")

# plt.title(f"Expected number of leaves: {sum(df["p_tree"] * df["n_leaves"]):.4f}")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
