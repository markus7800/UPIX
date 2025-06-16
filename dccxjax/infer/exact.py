
import jax
import jax._src.core as jax_core
from typing import List, Tuple, Any, Optional, Callable, TypeVar, Set, Dict, cast
from functools import reduce
from ..core.samplecontext import SampleContext
from ..core.model_slp import Model, SLP
from ..core.branching_tracer import retrace_branching_decisions
from ..types import Trace, FloatArray, PRNGKey, FloatArrayLike
from dccxjax.distributions import Distribution, DIST_SUPPORT, DIST_SUPPORT_LIKE
import jax.numpy as jnp

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
        self.log_probs: Dict[str,Tuple[FloatArray,bool]] = dict()
        self.rng_key = rng_key
    def sample(self, address: str, distribution: Distribution[DIST_SUPPORT, DIST_SUPPORT_LIKE], observed: Optional[DIST_SUPPORT_LIKE] = None) -> DIST_SUPPORT:
        if observed is not None:
            _observed = cast(DIST_SUPPORT, observed)
            self.log_probs[address] = (distribution.log_prob(_observed), True)
            return _observed
        assert distribution.numpyro_base._validate_args
        if address in self.X:
            value = cast(DIST_SUPPORT, self.X[address])
        else:
            self.rng_key, sample_key = jax.random.split(self.rng_key)
            value = distribution.sample(sample_key)
        self.log_probs[address] = (value, False)
        return value
    def logfactor(self, lf: FloatArrayLike, address: str = "__log_factor__") -> None:
        self.log_probs[address] = (self.log_probs.get(address, jnp.array(0.,float))[0] + lf, True)
    
    
from pprint import pprint

def get_factors(slp: SLP):
    
    def _m(_X: Dict) -> Dict[str,Tuple[FloatArray,bool]]:
        with FactorsCtx(_X) as ctx:
            slp.model()
        return ctx.log_probs
    
    X = slp.decision_representative
    
    with jax_core.take_current_trace() as parent_trace:
        trace = NamedTrace(parent_trace)
        with jax_core.set_current_trace(trace):
            in_tracers = {addr: NamedTracer(trace, val, {addr}) for addr, val in X.items()}
            
            _m_retraced = retrace_branching_decisions(_m, slp.branching_decisions)
            out_tracers, path_decision_tracers = _m_retraced(in_tracers)
            out = {addr: (tracer.val, is_observed, tracer.names) if isinstance(tracer, NamedTracer) else (tracer, is_observed) for addr, (tracer, is_observed) in out_tracers.items()}
            path_decisions = [(tracer.val, tracer.names) if isinstance(tracer, NamedTracer) else tracer for tracer in path_decision_tracers]
            pprint(out)
            pprint(path_decisions)
        