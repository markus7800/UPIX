import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
import numpy as np
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
    slope_xs = a / b
    intercept_xs = -slope_xs * xs.min() + 0
    xs = slope_xs * xs + intercept_xs
    xs_val = slope_xs * xs_val + intercept_xs
    rescale_x = lambda x: pd.to_datetime((np.array(x) - intercept_xs) / slope_xs)

    a = ys.max() - ys.min()
    slope_ys = 1 / a
    intercept_ys = -ys.mean() / a
    ys = slope_ys * ys + intercept_ys
    ys_val = slope_ys * ys_val + intercept_ys
    rescale_y = lambda x: (np.array(x) - intercept_ys) / slope_ys


    return jnp.array(xs), jnp.array(xs_val), jnp.array(ys), jnp.array(ys_val), rescale_x, rescale_y

xs_autogp, xs_val_autogp, ys_autogp, ys_val_autogp, rescale_x_autogp, rescale_y_autogp = get_data_autogp()

# plt.scatter(xs, ys, s=2)
# plt.scatter(xs_val, ys_val, s=2)
# plt.show()
# %%
# plt.scatter(xs_sdvi, ys_sdvi, s=2)
# plt.scatter(xs_val_sdvi, ys_val_sdvi, s=2)
# plt.scatter(xs_autogp, ys_autogp, s=2)
# plt.scatter(xs_val_autogp, ys_val_autogp, s=2)
# %%