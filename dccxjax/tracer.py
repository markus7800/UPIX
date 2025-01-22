import jax._src as jax_src
import jax._src.core as jax_core
import jax._src.interpreters.ad as jax_ad
import jax
from typing import List

# based on JVPTrace / JVPTracer

class BranchingTracer(jax_core.Tracer):
    
    def __init__(self, trace: jax_core.Trace, val):
        print("Init BranchingTracer with", val)
        self._trace = trace
        self.val = val

    @property
    def aval(self):
        return jax_core.get_aval(self.val)
    
    # def to_concrete_value(self):
    #     return jax_core.to_concrete_value(self.val)

    # def __bool__(self):
    #     return True

    # def __index__(self):
    #     assert False

    def full_lower(self):
        return jax_core.full_lower(self.val)
    
# op(tracer) -> op(tracer.aval)
# __getattr__(tracer, name)
#  = getattr(tracer.aval: ShapedArray, name)
#  = setattr(shaped_array, f"_{operator_name}", staticmethod(function)) for operator_name, function in _array_operators.items() in def _set_shaped_array_attributes(shaped_array):
#  = _defer_to_unrecognized_arg(op_symbol, ufunc.op) # Ensure that other array types have the chance to override arithmetic.
#  = @export @partial(jit, inline=True) def op(x: ArrayLike, /) -> Array: ...



class BranchingTrace(jax_core.Trace):
    def __init__(self, parent_trace) -> None:
        self.parent_trace = parent_trace

    def process_primitive(self, primitive: jax_core.Primitive, tracers, params):
        print("process_primitive", primitive, tracers)#, params)
        args = [tracer.val if isinstance(tracer, BranchingTracer) else tracer for tracer in tracers] # TODO: proper lowering
        out = primitive.bind_with_trace(self.parent_trace, args, params)
        if primitive.multiple_results:
            out = [BranchingTracer(self, o) for o in out]
        else:
            out = BranchingTracer(self, out)
        return out


def detect_branching(f):
    def _f(*args):
        with jax_core.take_current_trace() as parent_trace:
            print("parent_trace:", parent_trace)
            trace = BranchingTrace(parent_trace)
            with jax_core.set_current_trace(trace):
                in_tracers: List[BranchingTracer] = [BranchingTracer(trace, arg) for arg in args]
                out = f(*in_tracers)
                if isinstance(out, BranchingTracer):
                    return out.val
                else:
                    assert isinstance(out, list)
                    assert (all(map(lambda o: isinstance(o, BranchingTracer), out)))
                    return [o.val for o in out]
    return _f
