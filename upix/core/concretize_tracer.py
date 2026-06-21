import jax
import jax._src.core as jax_core
from typing import List, Tuple, Any, Callable, TypeVar, ParamSpec

__all__ = [
    "concretize"
]

class Decisions:
    def __init__(self) -> None:
        self.decisions: List[Any] = []
        
    def __len__(self) -> int:
        return len(self.decisions)
    
    def __getitem__(self, i):
        return self.decisions[i]
    
    def append(self, el):
        self.decisions.append(el)
    
    def to_human_readable(self) -> str:
        if len(self.decisions) == 0:
            return "Decisions(None.)"
        else:
            return "Decisions(" + ", ".join(f"{i}. {val}" for i, val in enumerate(self.decisions)) + ")"
        
    def __repr__(self) -> str:
        return repr(self.decisions)
        

# based on JVPTrace / JVPTracer

class ConcretizeTracer(jax_core.Tracer):
    
    def __init__(self, trace: jax_core.Trace, val):
        assert isinstance(trace, ConcretizeTrace)
        # print("ConcretizeTracer", id(self), type(val), id(val))
        # print(trace.object_id_to_name)
        self._trace = trace
        self.val = val

    @property
    def aval(self):
        return jax_core.get_aval(self.val)
    
    # def to_concrete_value(self):
    #     return jax_core.to_concrete_value(self.val)

    def _concretize(self):
        assert isinstance(self._trace, ConcretizeTrace)
        decisions = self._trace.decisions
        if self._trace.retrace:
            b = decisions[self._trace.decision_cnt]
            self._trace.decision_cnt += 1

            # self.val is tracer of parent trace
            with jax_core.set_current_trace(self._trace.parent_trace):
                #print(f"{self.val=} {concrete_val=} {self.val == concrete_val} {self._trace.path_condition=}")
                # print(type(self.val))
                # print(self.val == b)
                self._trace.path_condition = self._trace.path_condition & (self.val == b)
                self._trace.path_decisions.append(self.val == b)
            return b
        else:
            concrete_val = jax_core.to_concrete_value(self.val)
            assert concrete_val is not None
            b = concrete_val
            decisions.append(b)

            return b

    def __bool__(self):
        return bool(self._concretize())
        
    def __int__(self):
        return int(self._concretize())
    
    def __float__(self):
        return float(self._concretize())
    
    def __complex__(self):
        return complex(self._concretize())
    
    def __hex__(self):
        return hex(self._concretize())
    
    def __oct__(self):
        return oct(self._concretize())
    
    def __index__(self):
        return int(self._concretize())
    
    def item(self, *args):
        return self._concretize().item(*args)

    def full_lower(self):
        return jax_core.full_lower(self.val)
    

def concretize(a: jax.typing.ArrayLike) -> Any:
    if isinstance(a, ConcretizeTracer):
        return a._concretize()
    else:
        return a
    
# op(tracer) -> op(tracer.aval)
# setattr(tracer, f"__{operator_name}__", _forward_operator_to_aval(operator_name)) for operator_name in _array_operators in _set_tracer_aval_forwarding(tracer)


# __getattr__(tracer, name)
#  = getattr(tracer.aval: ShapedArray, name)
#  = setattr(shaped_array, f"_{operator_name}", staticmethod(function)) for operator_name, function in _array_operators.items() in def _set_shaped_array_attributes(shaped_array):
#  = _defer_to_unrecognized_arg(op_symbol, ufunc.op) # Ensure that other array types have the chance to override arithmetic.
#  = @export @partial(jit, inline=True) def op(x: ArrayLike, /) -> Array: ...

def maybe_concretize_tracer(trace: "ConcretizeTrace", val):
    try:
        # use JAXs implementation to determine if val is abstract array
        _ = jax_core.get_aval(val)
        return ConcretizeTracer(trace, val)
    except TypeError:
        return val
    
    # if isinstance(val, jax.Array):
    #     return ConcretizeTracer(trace, val, sexpr)
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

class ConcretizeTrace(jax_core.Trace):
    def __init__(self, parent_trace, decisions: Decisions, retrace: bool) -> None:
        super().__init__()
        # print("parent_trace of", self, "is", parent_trace)
        self.parent_trace = parent_trace
        self.decisions = decisions
        self.path_condition = True
        self.path_decisions: List[bool] = []
        self.retrace = retrace
        self.decision_cnt = 0

    def process_primitive(self, primitive: jax_core.Primitive, tracers, params):
        # print("process_primitive", primitive_name(primitive, params), tracers)
        # print(params)
        args = [tracer.val if isinstance(tracer, ConcretizeTracer) else tracer for tracer in tracers]
        # print("args =", args)
        out = primitive.bind_with_trace(self.parent_trace, args, params)
        if primitive.multiple_results:
            out_tracer = [maybe_concretize_tracer(self, o) for o in out]
            # print("outm =", out, out_tracer)
        else:
            out_tracer = maybe_concretize_tracer(self, out)
            # print("out1 =", out, out_tracer)
        return out_tracer
    
    def process_custom_jvp_call(self, primitive: jax_core.Primitive, fun, jvp, tracers, *, symbolic_zeros):
        # print(primitive)
        # print(fun)
        # print(jvp)
        args = [tracer.val if isinstance(tracer, ConcretizeTracer) else tracer for tracer in tracers]
        params = dict(symbolic_zeros=symbolic_zeros)
        out = primitive.bind_with_trace(self.parent_trace, (fun, jvp) + tuple(args), params)
        assert primitive.multiple_results
        out_tracer = [maybe_concretize_tracer(self, o) for o in out]
        return out_tracer


RET_TYPE = TypeVar("RET_TYPE")
FUNC_PARAM_SPEC = ParamSpec("FUNC_PARAM_SPEC")
# FUNC_TYPE = TypeVar("FUNC_TYPE", bound=Callable)

def execute_tracing_with_trace(trace: ConcretizeTrace, f: Callable[..., RET_TYPE], args) -> RET_TYPE:
    with jax_core.set_current_trace(trace):
        # in_flat, in_tree = jax.tree.flatten(args)
        # in_flat = map(lambda x: maybe_concretize_tracer(trace, x), in_flat)
        # in_tracers = jax.tree.unflatten(in_tree, in_flat)
        in_tracers = jax.tree.map(lambda x: maybe_concretize_tracer(trace, x), args)
        out_tracers = f(*in_tracers)
        # out_flat, out_tree = jax.tree.flatten(out_tracers)
        # out_flat = list(map(lambda x: x.val if isinstance(x, ConcretizeTracer) else x, out_flat))
        # out = jax.tree.unflatten(out_tree, out_flat)
        out = jax.tree.map(lambda x:  x.val if isinstance(x, ConcretizeTracer) else x, out_tracers)
        return out

def replay_decisions(f: Callable[FUNC_PARAM_SPEC, RET_TYPE], decisions: Decisions) -> Callable[FUNC_PARAM_SPEC, Tuple[RET_TYPE,bool]]:
    def _f(*args, **kwargs) -> Tuple[RET_TYPE,bool]:
        assert len(kwargs) == 0
        with jax_core.take_current_trace() as parent_trace:
            trace = ConcretizeTrace(parent_trace, decisions, retrace=True)
            out = execute_tracing_with_trace(trace, f, args)
            return out, trace.path_condition
    return _f

def replay_decisions_2(f: Callable[FUNC_PARAM_SPEC, RET_TYPE], decisions: Decisions) -> Callable[FUNC_PARAM_SPEC, Tuple[RET_TYPE,List[bool]]]:
    def _f(*args, **kwargs) -> Tuple[RET_TYPE,List[bool]]:
        assert len(kwargs) == 0
        with jax_core.take_current_trace() as parent_trace:
            trace = ConcretizeTrace(parent_trace, decisions, retrace=True)
            out = execute_tracing_with_trace(trace, f, args)
            return out, trace.path_decisions
    return _f

def track_decisions(f: Callable[FUNC_PARAM_SPEC, RET_TYPE]) -> Callable[FUNC_PARAM_SPEC, Tuple[RET_TYPE,Decisions]]:
    def _f(*args, **kwargs) -> Tuple[RET_TYPE,Decisions]:
        assert len(kwargs) == 0
        decisions = Decisions()
        with jax_core.take_current_trace() as parent_trace:
            trace = ConcretizeTrace(parent_trace, decisions, retrace=False)
            out = execute_tracing_with_trace(trace, f, args)
            return out, decisions
    return _f