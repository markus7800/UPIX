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
    
    def to_concrete_value(self):
        return self.val # jax_core.to_concrete_value(self.val)

    def __bool__(self):
        return True

    def __index__(self):
        assert False

# op(tracer) -> op(tracer.aval)
# __getattr__(tracer, name)
#  = getattr(tracer.aval: ShapedArray, name)
#  = setattr(shaped_array, f"_{operator_name}", staticmethod(function)) for operator_name, function in _array_operators.items() in def _set_shaped_array_attributes(shaped_array):
#  = _defer_to_unrecognized_arg(op_symbol, ufunc.op) # Ensure that other array types have the chance to override arithmetic.
#  = @export @partial(jit, inline=True) def op(x: ArrayLike, /) -> Array: ...



class BranchingTrace(jax_core.Trace):
    def __init__(self, parent_trace) -> None:
        self.parent_trace = parent_trace

    def process_primitive(self, primitive, tracers, params):
        print("process_primitive", primitive, tracers)#, params)
        # assert all(isinstance(tracer, BranchingTracer) for tracer in tracers)
        args = [tracer.val if isinstance(tracer, BranchingTracer) else tracer for tracer in tracers]
        print("args =", args)
        # out = primitive.impl(*args, **params)
        # print("jvp =", jax_ad.primitive_jvps[primitive], hash(primitive))
        out = primitive.bind_with_trace(self.parent_trace, args, params)
        print("out =", out, type(out))
        if primitive.multiple_results:
            out = [BranchingTracer(self, o) for o in out]
            print("out =", out)
            print("out concrete =", [o.to_concrete_value() for o in out])
        else:
            out = BranchingTracer(self, out)
            print("out =", out)
            print("out concrete =", out.to_concrete_value())
        return out


def detect_branching(f):
    def _f(*args):
        with jax_core.take_current_trace() as parent_trace:
            print("parent_trace:", parent_trace)
            trace = BranchingTrace(parent_trace)
            with jax_core.set_current_trace(trace):
                in_tracers: List[BranchingTracer] = [BranchingTracer(trace, arg) for arg in args]
                out = f(*in_tracers)
                return out
    return _f
