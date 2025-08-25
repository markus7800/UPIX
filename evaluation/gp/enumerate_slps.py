
from typing import Tuple, List
import itertools
from dccxjax.core import *
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

def find_active_slps_through_enumeration(
    N_LEAF_NODE_TYPES: int, active_slps: List[SLP], rng_key: PRNGKey, n_slps: int, model: Model,
    max_n_leaf = 4):
    
    class Counter:
        def __init__(self) -> None:
            self.c = 0
        def inc(self):
            self.c += 1
            return self.c
    def _is_valid(ts: Tuple[int,...], idx: int, c: Counter):
        i = c.inc()
        if i > len(ts):
            return False, 0
        if ts[i - 1] < N_LEAF_NODE_TYPES:
            return True, i
        else:
            v1, i1 = _is_valid(ts, 2*idx, c)
            v2, i2 = _is_valid(ts, 2*idx+1, c)
            return v1 and v2, max(i1,i2)
    valid_ts = []
    for ts in itertools.product(*([range(N_LEAF_NODE_TYPES+2)]*(2*max_n_leaf-1))):
        is_valid, size = _is_valid(ts, 1, Counter())
        if is_valid and size <= 2*max_n_leaf - 1:
            valid_ts.append(ts[:size])
            # print(ts, ts[:size])
    
    valid_ts = list(dict.fromkeys(valid_ts)) # remove duplicates
    valid_ts = sorted(valid_ts, key=lambda ts: (len(ts), ts))
    # print(len(valid_ts))
    
    def get_y(ts: Tuple[int,...], idx: int, c: Counter, Y: Trace):
        i = c.inc()
        Y[f"{idx}_node_type"] = jnp.array(ts[i - 1], int)
        if ts[i - 1] >= N_LEAF_NODE_TYPES:
            get_y(ts, 2*idx, c, Y)
            get_y(ts, 2*idx+1, c, Y)
        return Y
    
    for ts in valid_ts:
        rng_key, generate_key = jax.random.split(rng_key)
        trace, _ = model.generate(generate_key, get_y(ts, 1, Counter(), dict()))
        if model.equivalence_map is not None:
            trace = model.equivalence_map(trace)
        if all(slp.path_indicator(trace) == 0 for slp in active_slps):
            slp = slp_from_decision_representative(model, trace)
            active_slps.append(slp)
            tqdm.write(f"Discovered SLP {slp.formatted()}.")
        if len(active_slps) == n_slps:
            break
        
    print(f"{len(active_slps)=}")
    # exit(0)