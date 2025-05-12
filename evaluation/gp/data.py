#%%
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp

# df_airline = pd.read_csv("evaluation/gp/airline.csv", header=None)

# df_tsdl_161 = pd.read_csv("evaluation/gp/tsdl.161.csv", header=None)

# plt.plot(df_airline[0], df_airline[1])
# n_years = 1960 - 1949 + 1
# plt.plot(jnp.arange(0,n_years,1/12), df_tsdl_161[1] - df_tsdl_161[1].mean())
# plt.show()
# %%
def get_data_sdvi():
    df_airline = pd.read_csv("evaluation/gp/airline.csv", header=None)
    data = df_airline
    xs = data[0]
    ys = data[1]
    xs -= xs.min()
    xs /= xs.max()
    ys -= ys.mean()
    ys *= 4 / (ys.max() - ys.min())

    val_ix = round(xs.shape[0] * 0.9)
    xs, xs_val = xs[:val_ix], xs[val_ix:]
    ys, ys_val = ys[:val_ix], ys[val_ix:]

    return jnp.array(xs), jnp.array(xs_val), jnp.array(ys), jnp.array(ys_val)

xs_sdvi, xs_val_sdvi, ys_sdvi, ys_val_sdvi = get_data_sdvi()
# plt.scatter(xs, ys, s=2)
# plt.scatter(xs_val, ys_val, s=2)
# plt.show()
# %%
def get_data_autogp():
    df_tsdl_161 = pd.read_csv("evaluation/gp/tsdl.161.csv", header=None)
    data = df_tsdl_161

    xs = pd.to_numeric(pd.to_datetime(data[0]))
    ys = data[1]

    xs, xs_val = xs[:-18], xs[-18:]
    ys, ys_val = ys[:-18], ys[-18:]

    a = 1 - 0
    b = xs.max() - xs.min()
    slope = a / b
    intercept = -slope * xs.min() + 0
    xs = slope * xs + intercept
    xs_val = slope * xs_val + intercept

    a = ys.max() - ys.min()
    slope = 1 / a
    intercept = -ys.mean() / a
    ys = slope * ys + intercept
    ys_val = slope * ys_val + intercept

    return jnp.array(xs), jnp.array(xs_val), jnp.array(ys), jnp.array(ys_val)

xs_autogp, xs_val_autogp, ys_autogp, ys_val_autogp = get_data_autogp()

# plt.scatter(xs, ys, s=2)
# plt.scatter(xs_val, ys_val, s=2)
# plt.show()
# %%
# plt.scatter(xs_sdvi, ys_sdvi, s=2)
# plt.scatter(xs_val_sdvi, ys_val_sdvi, s=2)
# plt.scatter(xs_autogp, ys_autogp, s=2)
# plt.scatter(xs_val_autogp, ys_val_autogp, s=2)
# %%