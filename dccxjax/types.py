import jax
from typing import Dict, Union, NamedTuple
from dataclasses import dataclass
from .utils import broadcast_jaxtree
from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "PRNGKey",
    "Trace",
    "Traces",
    "StackedTrace",
    "StackedTraces",
    "to_traces",
    "ArrayLike",
    "BoolArrayLike",
    "IntArrayLike",
    "FloatArrayLike",
    "BoolArray",
    "IntArray",
    "FloatArray",
]

# StaticScalar = Union[
#   np.bool_, np.number,  # NumPy scalar types
#   bool, int, float, complex,  # Python scalar types
# ]

# ArrayLike = Union[
#   Array,  # JAX array type
#   np.ndarray,  # NumPy array type
#   StaticScalar,  # valid scalars
# ]

# just for annotation
BoolArray = Array
IntArray = Array
FloatArray = Array

BoolArrayLike = bool | Array
IntArrayLike = int | Array
FloatArrayLike = float | Array



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

# data[address] has shape (N,T,D) where D is the dimensionality of the RV
@dataclass
class StackedTraces:
    data: Trace
    N: int # n_samples_per_chain
    T: int # n_chains
    def n_samples(self):
        return self.T * self.N
    def unstack(self) -> Traces:
        return Traces(jax.tree_map(_unstack, self.data), self.N * self.T)
    def get(self, sample_ix, chain_ix) -> Trace:
        return jax.tree_map(lambda v: v[sample_ix,chain_ix,...], self.data)
    def get_stacked(self, sample_ix) -> StackedTrace:
        return StackedTrace(jax.tree_map(lambda v: v[sample_ix,...], self.data), self.T)
    def get_chains(self, chain_ix) -> Traces:
        return Traces(jax.tree_map(lambda v: v[:,chain_ix,...], self.data), self.N)

# Scalar = Union[float, int]
# Numeric = Union[jax.Array, Scalar]
