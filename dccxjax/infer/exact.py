
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
        self.log_probs[address] = value
        return value
    def logfactor(self, lf: FloatArrayLike, address: str = "__log_factor__") -> None:
        self.log_probs[address] = self.log_probs.get(address, jnp.array(0.,float))[0] + lf
    

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
    def logfactor(self, lf: FloatArrayLike, address: str = "__log_factor__") -> None:
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

@dataclass
class Factor:
    addresses: List[str]
    table: FloatArray

    def __repr__(self) -> str:
        return f"Factor({self.addresses}, {self.table.shape})"

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
        # print(names, factor_variable_supports)
        meshgrids = jnp.meshgrid(*factor_variable_supports)
        factor_shape = meshgrids[0].shape
        print(addresses, factor_shape)
        partial_X = {addr: meshgrid.reshape(-1) for addr, meshgrid in zip(addresses, meshgrids)}
        
        @jax.vmap
        def _factor_fn(_partial_X: Trace) -> FloatArray:
            return all_factors_fn(_partial_X)[i][0]
        factor_fn = jax.jit(_factor_fn) if jit else _factor_fn
        
        factor_table = factor_fn(partial_X).reshape(factor_shape)
        
        factors.append(Factor(addresses, factor_table))

    return factors
        
        