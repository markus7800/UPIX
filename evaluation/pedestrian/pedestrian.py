from upix.core import *
import jax
import jax.numpy as jnp
import upix.distributions as dist
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict

@model
def pedestrian():
    start = sample("start", dist.Uniform(0.,3.))
    position = start
    distance = 0.
    t = 0
    while (position > 0) & (distance < 10):
        t += 1
        step = sample(f"step_{t}", dist.Uniform(-1.,1.))
        position += step
        distance += jax.lax.abs(step)
    sample("obs", dist.Normal(distance, 0.1), observed=1.1)
    return start


def pedestrian_rec_walk(t: int, position: FloatArrayLike, distance: FloatArrayLike) -> FloatArrayLike:
    if (position > 0) & (distance < 10):
        step = sample(f"step_{t}", dist.Uniform(-1.,1.))
        position += step
        distance += jax.lax.abs(step)
        return pedestrian_rec_walk(t+1, position, distance)
    else:
        return distance

@model
def pedestrian_recursive():
    start = sample("start", dist.Uniform(0.,3.))
    distance = pedestrian_rec_walk(1, start, 0.)
    sample("obs", dist.Normal(distance, 0.1), observed=1.1)
    return start



def find_t_max(slp: SLP):
    t_max = 0
    for addr in slp.decision_representative.keys():
        if addr.startswith("step_"):
            t_max = max(t_max, int(addr[5:]))
    return t_max

def formatter(slp: SLP):
    t_max = find_t_max(slp)
    return f"#steps={t_max}"
