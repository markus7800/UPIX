import jax
from typing import Dict, Union, NamedTuple
from dataclasses import dataclass
from .utils import broadcast_jaxtree

__all__ = [
    "PRNGKey",
    "Trace",
    "Traces",
    "StackedTrace",
    "StackedTraces",
    "to_traces",
]


PRNGKey = jax.Array

# data[address] is either a scalar for a univariate RV or an array for a multi-dimensional RV has shape (D,)
Trace = Dict[str, jax.Array]

def to_shaped_array_trace(X: Trace):
    return {address: value.aval for address, value in X.items()}

# data[address] has shape (N,D) where D is the dimensionality of the RV
@dataclass
class Traces:
    data: Trace
    N: int
    def n_samples(self):
        return self.N

def to_traces(X: Trace) -> Traces:
    return Traces(broadcast_jaxtree(X, (1,)), 1)

# data[address] has shape (T,D) where D is the dimensionality of the RV
@dataclass
class StackedTrace:
    data: Trace
    T: int
    def n_samples(self):
        return self.T
    
    def unstack(self) -> Traces:
        return Traces(self.data, self.T)
    
    def to_stacked_traces(self) -> "StackedTraces":
        return StackedTraces(jax.tree_map(lambda x: x.reshape((self.T, 1) + x.shape[1:]), self.data) , 1, self.T)

def _unstack(values: jax.Array):
    shape = values.shape
    assert len(shape) >= 2
    var_dim = () if len(shape) < 3 else (shape[2],)
    n_samples = shape[0]
    n_chains = shape[1]
    return jax.lax.reshape(values, (n_samples * n_chains, *var_dim))

# data[address] has shape (T,N,D) where D is the dimensionality of the RV
@dataclass
class StackedTraces:
    data: Trace
    N: int
    T: int
    def n_samples(self):
        return self.T * self.N
    def unstack(self) -> Traces:
        return Traces(jax.tree_map(_unstack, self.data), self.N * self.T)
    def get(self, n, t) -> Trace:
        return jax.tree_map(lambda v: v[n,t,...], self.data)

# Scalar = Union[float, int]
# Numeric = Union[jax.Array, Scalar]
