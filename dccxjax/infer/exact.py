
import jax
import jax._src.core as jax_core
from typing import List, Tuple, Any, Optional, Callable, TypeVar, Set, Dict, cast
from functools import reduce
from ..core.samplecontext import SampleContext
from ..core.model_slp import Model, SLP
from ..core.branching_tracer import retrace_branching_decisions
from ..types import Trace, FloatArray, PRNGKey, FloatArrayLike, IntArray
from dccxjax.distributions import Distribution, DIST_SUPPORT, DIST_SUPPORT_LIKE
import jax.numpy as jnp
import numpyro.distributions as numpyro_dist
from dataclasses import dataclass

__all__ = [
    
]


class NamedTracer(jax_core.Tracer):
    def __init__(self, trace: jax_core.Trace, val, names: Set[str]):
        assert isinstance(trace, NamedTrace)
        self._trace = trace
        self.val = val
        self.names: Set[str] = set() | names
        
    @property
    def aval(self):
        return jax_core.get_aval(self.val)

def maybe_named_tracer(trace: "NamedTrace", val, names):
    try:
        # use JAXs implementation to determine if val is abstract array
        _ = jax_core.get_aval(val)
        return NamedTracer(trace, val, names)
    except TypeError:
        return val

class NamedTrace(jax_core.Trace):
    def __init__(self, parent_trace) -> None:
        super().__init__()
        self.parent_trace = parent_trace
    def process_primitive(self, primitive: jax_core.Primitive, tracers, params):
        args = [tracer.val if isinstance(tracer, NamedTracer) else tracer for tracer in tracers]
        input_names: List[Set[str]] = [tracer.names for tracer in tracers if isinstance(tracer, NamedTracer)]
        output_names = reduce(lambda x, y: x | y, input_names, set())

        out = primitive.bind_with_trace(self.parent_trace, args, params)
        if primitive.multiple_results:
            out_tracer = [maybe_named_tracer(self, o, output_names) for o in out]
        else:
            out_tracer = maybe_named_tracer(self, out, output_names)
        return out_tracer
    def process_custom_jvp_call(self, primitive: jax_core.Primitive, fun, jvp, tracers, *, symbolic_zeros):
        args = [tracer.val if isinstance(tracer, NamedTracer) else tracer for tracer in tracers]
        input_names: List[Set[str]] = [tracer.names for tracer in tracers if isinstance(tracer, NamedTracer)]
        output_names = reduce(lambda x, y: x | y, input_names, set())
        params = dict(symbolic_zeros=symbolic_zeros)
        out = primitive.bind_with_trace(self.parent_trace, (fun, jvp) + tuple(args), params)
        assert primitive.multiple_results
        out_tracer = [maybe_named_tracer(self, o, output_names) for o in out]
        return out_tracer

    
class FactorsCtx(SampleContext):
    # X can be partial
    def __init__(self, X: Trace, rng_key: PRNGKey = jax.random.PRNGKey(0)) -> None:
        super().__init__()
        self.X = X
        self.log_probs: Dict[str,FloatArray] = dict()
        self.rng_key = rng_key
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            _observed = cast(DIST_SUPPORT, observed)
            self.log_probs[address] = distribution.log_prob(_observed)
            return _observed
        assert distribution.numpyro_base._validate_args
        if address in self.X:
            value = cast(DIST_SUPPORT, self.X[address])
        else:
            self.rng_key, sample_key = jax.random.split(self.rng_key)
            value = distribution.sample(sample_key)
        self.log_probs[address] = distribution.log_prob(value)
        return value
    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        self.log_probs[address] = self.log_probs.get(address, jnp.array(0.,float)) + lf
    

class SupportCtx(SampleContext):
    # X can be partial
    def __init__(self, X: Trace) -> None:
        super().__init__()
        self.X = X
        self.supports: Dict[str,Optional[IntArray]] = dict()
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            _observed = cast(DIST_SUPPORT, observed)
            return _observed
        if distribution.numpyro_base.has_enumerate_support:
            self.supports[address] = distribution.numpyro_base.enumerate_support()
        else:
            self.supports[address] = None    
        value = cast(DIST_SUPPORT, self.X[address])
        return value
    def logfactor(self, lf: FloatArrayLike, address: str) -> None:
        pass
    
def get_supports(slp: SLP) -> Dict[str,Optional[IntArray]]:
    with SupportCtx(slp.decision_representative) as ctx:
        slp.model()
        return ctx.supports

def make_all_factors_fn(slp: SLP):
    
    def _fs(_X: Dict) -> Dict[str,FloatArray]:
        with FactorsCtx(_X) as ctx:
            slp.model()
        return ctx.log_probs
    
    def _all_factors_fn(_X: Trace) -> List[Tuple[FloatArray,List[str]]]:
        with jax_core.take_current_trace() as parent_trace:
            trace = NamedTrace(parent_trace)
            with jax_core.set_current_trace(trace):
                in_tracers = {addr: NamedTracer(trace, val, {addr}) for addr, val in _X.items()}
                
                _fs_retraced = retrace_branching_decisions(_fs, slp.branching_decisions)
                out_tracers, path_decision_tracers = _fs_retraced(in_tracers)
                path_decision_tracers = [jax.lax.select(tracer, jnp.array(0.,float), jnp.array(-jnp.inf,float)) for tracer in path_decision_tracers]
                
                out = [(tracer.val, sorted(tracer.names)) if isinstance(tracer, NamedTracer) else (tracer, list()) for addr, tracer in out_tracers.items()]
                path_decisions = [(tracer.val, sorted(tracer.names)) if isinstance(tracer, NamedTracer) else tracer for tracer in path_decision_tracers]
                return out + path_decisions
            
    return _all_factors_fn

def issorted(l: List):
   return all(l[i] <= l[i+1] for i in range(len(l) - 1))

class Factor:
    addresses: List[str]
    table: FloatArray
    def __init__(self, addresses: List[str], table: FloatArray) -> None:
        assert issorted(addresses)
        if table is not None:
            assert len(table.shape) == len(addresses), f"{addresses} vs {table.shape}"
        self.addresses = addresses
        self.table = table
    def __repr__(self) -> str:
        assert issorted(self.addresses)
        return f"Factor({self.addresses}, {self.table.shape})"
    
def factor_flatten(v):
    children = (v.table,)
    aux_data = v.addresses
    return (children, aux_data)
def factor_unflatten(aux_data, children):
    return Factor(aux_data, *children)
from jax.tree_util import register_pytree_node
register_pytree_node(Factor, factor_flatten, factor_unflatten)
    
def factor_product(A: Factor, B: Factor) -> Factor:
    i = 0
    j = 0
    a_shape = A.table.shape
    b_shape = B.table.shape
    a_new_shape: List[int] = []
    b_new_shape: List[int] = []
    c_addresses: List[str] = []
    
    while i < len(A.addresses) and j < len(B.addresses):
        a_addr = A.addresses[i]
        b_addr = B.addresses[j]
        if a_addr == b_addr:
            a_new_shape.append(a_shape[i])
            b_new_shape.append(b_shape[j])
            c_addresses.append(a_addr)
            i += 1
            j += 1
        elif a_addr < b_addr:
            a_new_shape.append(a_shape[i])
            b_new_shape.append(1)
            c_addresses.append(a_addr)
            i += 1
        else: # a_addr > b_addr:
            a_new_shape.append(1)
            b_new_shape.append(b_shape[j])
            c_addresses.append(b_addr)
            j += 1
    while i < len(A.addresses):
        a_addr = A.addresses[i]
        a_new_shape.append(a_shape[i])
        b_new_shape.append(1)
        c_addresses.append(a_addr)
        i += 1
    while j < len(B.addresses):
        b_addr = B.addresses[j]
        a_new_shape.append(1)
        b_new_shape.append(b_shape[j])
        c_addresses.append(b_addr)
        j += 1
        
    c_table = A.table.reshape(a_new_shape) + B.table.reshape(b_new_shape)
    return Factor(c_addresses, c_table)

def factor_sum(A: Factor, addresses_to_sum_out: List[str]) -> Factor:
    axis = [i for i, addr in enumerate(A.addresses) if addr in addresses_to_sum_out]
    variables = [addr for i, addr in enumerate(A.addresses) if addr not in addresses_to_sum_out]
    table = jax.scipy.special.logsumexp(A.table, axis=axis)
    return Factor(variables, table)

def factor_sum_out_addr(A: Factor, address_to_sum_out: str) -> Factor:
    axis = [i for i, addr in enumerate(A.addresses) if addr == address_to_sum_out]
    variables = [addr for i, addr in enumerate(A.addresses) if addr != address_to_sum_out]
    table = jax.scipy.special.logsumexp(A.table, axis=axis)
    return Factor(variables, table)
  
def compute_factors(slp: SLP, jit: bool = True):
    all_factors_fn = make_all_factors_fn(slp)
    factor_prototypes = all_factors_fn(slp.decision_representative)
    supports = get_supports(slp)
    def _get_support(addr: str) -> IntArray:
        s = supports[addr]
        if s is None:
            return jnp.array([slp.decision_representative[addr]])
        else:
            return s
    
    factors: List[Factor] = []
    for i, (_, addresses) in enumerate(factor_prototypes):
        assert len(addresses) > 0
        factor_variable_supports: List[IntArray] = list(map(_get_support, addresses))
        # print(addresses, factor_variable_supports)
        meshgrids = jnp.meshgrid(*factor_variable_supports, indexing="ij")
        factor_shape = meshgrids[0].shape
        partial_X = {addr: meshgrid.reshape(-1) for addr, meshgrid in zip(addresses, meshgrids)}
        
        @jax.vmap
        def _factor_fn(_partial_X: Trace) -> FloatArray:
            return all_factors_fn(_partial_X)[i][0]
        factor_fn = jax.jit(_factor_fn) if jit else _factor_fn
        
        factor_table = factor_fn(partial_X).reshape(factor_shape)
        factor = Factor(addresses, factor_table)
        # print(factor)
        factors.append(factor)

    return factors
    
from pprint import pprint
def variable_elimination(factors: List[Factor], elimination_order: List[str]):
    # print("\nvariable_elimination")
    variable_to_factors: Dict[str, Set[Factor]] = {}
    for factor in factors:
        for addr in factor.addresses:
            if addr not in variable_to_factors:
                variable_to_factors[addr] = set()
            variable_to_factors[addr].add(factor)
            
    tau: Factor = Factor([], jnp.array(0.,float))
    for addr in elimination_order:
        # print(f"eliminate {addr}")
        # pprint(variable_to_factors)
        neighbour_factors = variable_to_factors[addr]
        assert len(neighbour_factors) > 0
        psi = reduce(factor_product, neighbour_factors)
        tau = factor_sum_out_addr(psi, addr)
        # print(f"{tau=}")
        for factor in neighbour_factors:
            for variable in factor.addresses:
                factor_set = variable_to_factors[variable]
                if factor_set is not neighbour_factors:
                    factor_set.discard(factor)
        for variable in tau.addresses:
            variable_to_factors[variable].add(tau)
        del variable_to_factors[addr]
        # print()
        
    if len(variable_to_factors) == 0:
        log_evidence = jax.scipy.special.logsumexp(tau.table)
        return Factor([], jnp.array(0.,float)), log_evidence
    else:
        remaining_factors = reduce(lambda x, y: x | y, variable_to_factors.values())
        result = reduce(factor_product, remaining_factors)
        log_evidence = jax.scipy.special.logsumexp(result.table)
        return result, log_evidence
        
    

        