import jax
from typing import Dict, Union, NamedTuple, TypeVar, Generic
from dataclasses import dataclass
from .utils import broadcast_jaxtree
from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "PRNGKey",
    "Trace",
    "SampleValues",
    "Traces",
    "StackedSampleValue",
    "StackedTrace",
    "StackedSampleValues",
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

VALUE_TYPE = TypeVar("VALUE_TYPE")

# data[address] has shape (N,D) where D is the dimensionality of the RV
@dataclass
class SampleValues(Generic[VALUE_TYPE]):
    data: VALUE_TYPE
    N: int
    def __repr__(self) -> str:
        return f"SampleValues({self.data.__class__.__name__}, {self.N:,})"
    def n_samples(self):
        return self.N
    def get(self) -> VALUE_TYPE:
        return self.data
Traces = SampleValues[Trace]

def to_traces(X: Trace) -> Traces:
    return Traces(broadcast_jaxtree(X, (1,)), 1)

# data[address] has shape (T,D) where D is the dimensionality of the RV
@dataclass
class StackedSampleValue(Generic[VALUE_TYPE]):
    data: VALUE_TYPE
    T: int
    def __repr__(self) -> str:
        return f"StackedSampleValue({self.data.__class__.__name__}, {self.T:,})"
    def n_samples(self):
        return self.T
    
    def unstack(self) -> SampleValues[VALUE_TYPE]:
        return SampleValues[VALUE_TYPE](self.data, self.T)
    
    def to_stacked_values(self) -> "StackedSampleValues[VALUE_TYPE]":
        return StackedSampleValues(jax.tree_map(lambda x: x.reshape((self.T, 1) + x.shape[1:]), self.data) , 1, self.T)
    
StackedTrace = StackedSampleValue[Trace]

# reshapes to remove second dimension (N,T,...) -> (N*T,...)
def _unstack_sample_data(values: jax.Array):
    shape = values.shape
    assert len(shape) >= 2
    var_dim = () if len(shape) < 3 else (shape[2],)
    n_samples = shape[0]
    n_chains = shape[1]
    return jax.lax.reshape(values, (n_samples * n_chains, *var_dim))

# data[address] has shape (N,T,D) where D is the dimensionality of the RV
@dataclass
class StackedSampleValues(Generic[VALUE_TYPE]):
    data: VALUE_TYPE
    N: int # n_samples_per_chain
    T: int # n_chains
    def __repr__(self) -> str:
        return f"StackedSampleValues({self.data.__class__.__name__}, {self.N:,} x {self.T:,})"
    def n_samples(self):
        return self.T * self.N
    def n_chains(self):
        return self.T
    def n_samples_per_chain(self):
        return self.N
    def unstack(self) -> SampleValues[VALUE_TYPE]:
        return SampleValues(jax.tree_map(_unstack_sample_data, self.data), self.N * self.T)
    def get(self) -> VALUE_TYPE:
        return self.data
    def get_selection(self, sample_ix, chain_ix) -> VALUE_TYPE:
        return jax.tree_map(lambda v: v[sample_ix,chain_ix,...], self.data)
    def get_stacked(self, sample_ix) -> StackedSampleValue[VALUE_TYPE]:
        return StackedSampleValue(jax.tree_map(lambda v: v[sample_ix,...], self.data), self.T)
    def get_chains(self, chain_ix) -> SampleValues[VALUE_TYPE]:
        return SampleValues(jax.tree_map(lambda v: v[:,chain_ix,...], self.data), self.N)
    def get_subset(self, sample_ix, chain_ix) -> "StackedSampleValues[VALUE_TYPE]":
        sub_data = self.get_selection(sample_ix, chain_ix)
        leafs, _ = jax.tree_flatten(sub_data)
        N, T = leafs[0].shape[0:2]
        return StackedSampleValues(sub_data, N, T)

StackedTraces = StackedSampleValues[Trace]

# Scalar = Union[float, int]
# Numeric = Union[jax.Array, Scalar]
