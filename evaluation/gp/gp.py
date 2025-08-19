import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from dccxjax.core import *
import dccxjax.distributions as dist
from kernels import *
from dataclasses import fields
from tqdm.auto import tqdm
from typing import Tuple, Optional, Dict

import logging
setup_logging(logging.WARN)

from data import *
xs, xs_val, ys, ys_val = get_data_autogp()


def normalise(a: jax.Array): return a / a.sum()

# AutoGP
# N_LEAF_NODE_TYPES = 5
# NODE_TYPES: List[type[GPKernel]] = [Constant, Linear, SquaredExponential, GammaExponential, Periodic, Plus, Times]
# NODE_TYPE_PROBS = normalise(jnp.array([0, 6, 0, 6, 6, 5, 5],float))

# Reichelt
N_LEAF_NODE_TYPES = 4
NODE_TYPES: List[type[GPKernel]] = [UnitRationalQuadratic, UnitPolynomialDegreeOne, UnitSquaredExponential, UnitPeriodic, Plus, Times]
NODE_TYPE_PROBS = normalise(jnp.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1],float))

def covariance_prior(idx: int) -> GPKernel:
    node_type = sample(f"{idx}_node_type", dist.Categorical(NODE_TYPE_PROBS))
    NodeType = NODE_TYPES[node_type]
    if node_type < N_LEAF_NODE_TYPES:
        params = []
        for field in fields(NodeType):
            field_name = field.name
            log_param = sample(f"{idx}_{field_name}", dist.Normal(0., 1.))
            param = transform_param(field_name, log_param)
            params.append(param)
        return NodeType(*params)
    else:
        left = covariance_prior(2*idx)
        right = covariance_prior(2*idx+1)
        return NodeType(left, right) # type: ignore
    
@model
def gaussian_process(xs: jax.Array, ts: jax.Array):
    kernel = covariance_prior(1)
    noise = sample("noise", dist.Normal(0.,1.))
    noise = transform_param("noise", noise) + 1e-5
    cov_matrix = kernel.eval_cov_vec(xs) + noise * jnp.eye(xs.size)
    # MultivariateNormal does cholesky internally
    sample("obs", dist.MultivariateNormal(jnp.zeros_like(xs), covariance_matrix=cov_matrix), observed=ts)

def _get_gp_kernel(trace: Trace, idx: int, ordered: bool) -> GPKernel:
    node_type = trace[f"{idx}_node_type"]
    if node_type < N_LEAF_NODE_TYPES:
        NodeType = NODE_TYPES[node_type]
        params = []
        for field in fields(NodeType):
            field_name = field.name
            log_param = trace[f"{idx}_{field_name}"]
            param = transform_param(field_name, log_param)
            params.append(param)
        return NodeType(*params)
    else:
        NodeType = [Plus, Times][node_type - N_LEAF_NODE_TYPES]
        # de-duplicate
        left = _get_gp_kernel(trace, 2*idx, ordered)
        right = _get_gp_kernel(trace, 2*idx+1, ordered)
        if ordered and left.name() > right.name():
            return NodeType(right, left)
        else:
            return NodeType(left, right)
def get_gp_kernel(trace: Trace, ordered: bool = True) -> GPKernel:
    return _get_gp_kernel(trace, 1, ordered)


def _equivalence_trace(old_trace: Trace, old_idx: int, new_trace: Trace, new_idx: int):
    node_type = old_trace[f"{old_idx}_node_type"]
    new_trace[f"{new_idx}_node_type"] = node_type
    if node_type < N_LEAF_NODE_TYPES:
        for field in fields(NODE_TYPES[node_type]):
            field_name = field.name
            new_trace[f"{new_idx}_{field_name}"] = old_trace[f"{old_idx}_{field_name}"]
    else:
        old_left_cls = NODE_TYPES[old_trace[f"{2*old_idx}_node_type"]]
        old_right_cls = NODE_TYPES[old_trace[f"{2*old_idx+1}_node_type"]]
        if old_left_cls.name() > old_right_cls.name():
            _equivalence_trace(old_trace, 2*old_idx+1, new_trace, 2*new_idx)
            _equivalence_trace(old_trace, 2*old_idx, new_trace, 2*new_idx+1)
        else:
            _equivalence_trace(old_trace, 2*old_idx, new_trace, 2*new_idx)
            _equivalence_trace(old_trace, 2*old_idx+1, new_trace, 2*new_idx+1)
            
def equivalence_map(trace: Trace) -> Trace:
    equivalence_class_representative: Trace = dict()
    _equivalence_trace(trace, 1, equivalence_class_representative, 1)
    if "noise" in trace:
        equivalence_class_representative["noise"] = trace["noise"]
    return equivalence_class_representative


