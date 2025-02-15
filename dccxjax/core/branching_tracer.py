import jax
import jax._src.core as jax_core
from jax.tree_util import tree_flatten, tree_unflatten
from typing import List, Tuple, Any, Optional, Callable
from .sexpr import SExpr, SConstant, SOp

__all_ = [
    "branching"
]

class BranchingDecisions:
    def __init__(self) -> None:
        self.decisions: List[Tuple[SExpr, Any]] = []

    def to_human_readable(self) -> str:
        expressions: List[str] = []
        for sexpr, val in self.decisions:
            expr = sexpr.to_human_readable()
            if isinstance(val, jax.Array) and val.dtype == jax.numpy.dtype("bool") and val.shape == ():
                expressions.append(expr if val.item() else "~"+expr)
            else:
                expressions.append(expr + " = " + SConstant(val).to_human_readable())
        if len(expressions) == 0:
            return "No decisions."
        else:
            return " and\n".join(expressions)
        

# based on JVPTrace / JVPTracer

class BranchingTracer(jax_core.Tracer):
    
    def __init__(self, trace: jax_core.Trace, val, sexpr: Optional[SExpr] = None):
        assert isinstance(trace, BranchingTrace)
        # print("BranchingTracer", id(self), type(val), id(val))
        # print(trace.object_id_to_name)
        self._trace = trace
        self.val = val
        self.sexpr = sexpr if sexpr is not None else SConstant(val)

    @property
    def aval(self):
        return jax_core.get_aval(self.val)
    
    # def to_concrete_value(self):
    #     return jax_core.to_concrete_value(self.val)

    def _branching(self):
        assert isinstance(self._trace, BranchingTrace)
        decisions = self._trace.branching_decisions.decisions
        if self._trace.retrace:
            _, b = decisions[self._trace.decision_cnt]
            self._trace.decision_cnt += 1
            return b
        else:
            concrete_val = jax_core.to_concrete_value(self.val)
            assert concrete_val is not None
            b = concrete_val
            decisions.append((self.sexpr, b))
            return b

    def __bool__(self):
        return bool(self._branching())
        
    def __index__(self):
        return int(self._branching())

    def full_lower(self):
        return jax_core.full_lower(self.val)
    

def branching(tracer: BranchingTracer):
    return tracer._branching()
    
# op(tracer) -> op(tracer.aval)
# setattr(tracer, f"__{operator_name}__", _forward_operator_to_aval(operator_name)) for operator_name in _array_operators in _set_tracer_aval_forwarding(tracer)


# __getattr__(tracer, name)
#  = getattr(tracer.aval: ShapedArray, name)
#  = setattr(shaped_array, f"_{operator_name}", staticmethod(function)) for operator_name, function in _array_operators.items() in def _set_shaped_array_attributes(shaped_array):
#  = _defer_to_unrecognized_arg(op_symbol, ufunc.op) # Ensure that other array types have the chance to override arithmetic.
#  = @export @partial(jit, inline=True) def op(x: ArrayLike, /) -> Array: ...

def maybe_branching_tracer(trace: "BranchingTrace", val, sexpr: Optional[SExpr] = None):
    try:
        jax_core.get_aval(val)
        return BranchingTracer(trace, val, sexpr)
    except TypeError:
        return val
    
    # if isinstance(val, jax.Array):
    #     return BranchingTracer(trace, val, sexpr)
    # else:
    #     return val

# jit(branch(f))
# jit is lower than branch, jit is parent of branch
# t1 = jit
# t2 = branch

# eval(t1(t2(f)))
# 2 t2_trace    
# 1 t1_trace    (parent is lower)
# 0 EvalTrace       
# -> t1 is parent of t2
# -> lower(t2_tracer) ~> t1_tracer
# t2_tracer(t1_tracer(val))
# -> process_primitive may call parent_trace.process_primitive
# -> t2_trace.process_primitive may call t1_trace.process_primitive

class BranchingTrace(jax_core.Trace):
    def __init__(self, parent_trace, branching_decisions: BranchingDecisions, retrace: bool) -> None:
        # print("parent_trace of", self, "is", parent_trace)
        self.parent_trace = parent_trace
        self.branching_decisions = branching_decisions
        self.retrace = retrace
        self.decision_cnt = 0

    def process_primitive(self, primitive: jax_core.Primitive, tracers, params):
        # print("process_primitive", primitive_name(primitive, params), tracers)
        # print(params)
        args = [tracer.val if isinstance(tracer, BranchingTracer) else tracer for tracer in tracers]
        # print("args =", args)
        sargs = [tracer.sexpr if isinstance(tracer, BranchingTracer) else SConstant(tracer) for tracer in tracers]
        # print("sargs =", sargs)
        out = primitive.bind_with_trace(self.parent_trace, args, params)
        sop = SOp(primitive, sargs, params)
        if primitive.multiple_results:
            out_tracer = [maybe_branching_tracer(self, o, sexpr=sop) for o in out]
            # print("outm =", out, out_tracer)
        else:
            out_tracer = maybe_branching_tracer(self, out, sexpr=sop)
            # print("out1 =", out, out_tracer)
        return out_tracer


def trace_branching(f: Callable, branching_decisions: BranchingDecisions, retrace: bool = False):
    def _f(*args):
        with jax_core.take_current_trace() as parent_trace:
            trace = BranchingTrace(parent_trace, branching_decisions, retrace)
            with jax_core.set_current_trace(trace):
                in_flat, in_tree = tree_flatten(args)
                in_flat = map(lambda x: maybe_branching_tracer(trace, x), in_flat)
                # print("in_tree =", in_tree)
                # print("in_flat =", in_flat)
                in_tracers = tree_unflatten(in_tree, in_flat)
                out = f(*in_tracers)
                out_flat, out_tree = tree_flatten(out)
                out_flat = list(map(lambda x: x.val if isinstance(x, BranchingTracer) else x, out_flat))
                # print("out_flat =", out_flat)
                return tree_unflatten(out_tree, out_flat)
    return _f
