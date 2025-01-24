import jax._src as jax_src
import jax._src.core as jax_core
import jax._src.interpreters.ad as jax_ad
import jax._src.pjit as jax_pjit
from jax._src.dtypes import _jax_types
import jax
from typing import List, Optional, Dict, Tuple, Any, Callable
from abc import ABC, abstractmethod
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp


def primitive_name(primitive: jax_core.Primitive, params):
    if primitive.name == "pjit":
        pjit_name = params["name"]
        return f"pjit<{pjit_name}>"
    else:
        return primitive.name
    
class SExpr(ABC):
    @abstractmethod
    def eval(self, X: dict[str, jax.Array]) -> jax.Array:
        raise NotImplementedError
    
class SOp(SExpr):
    def __init__(self, primitive: jax_core.Primitive, args: List[SExpr], params) -> None:
        self.primitive = primitive
        self.params = params
        self.args = args
    def __repr__(self) -> str:
        return primitive_name(self.primitive, self.params) + "(" + ", ".join(map(repr, self.args)) + ")"
    def eval(self, X: dict[str, jax.Array]) -> jax.Array:
        args = [arg.eval(X) for arg in self.args]
        # print("eval", primitive_name(self.primitive, self.params), "with", args, self.args, self.params)
        # out = self.primitive.impl(*args, **self.params)
        out = self.primitive.bind(*args, **self.params) # bind instead of impl is important here to be able to jit-compile it later
        # pjit always returns []
        if self.primitive.multiple_results:
            if self.primitive is jax_pjit.pjit_p and self.params["out_layouts"] == (None,):
                # print("out1jit =", out)
                return out[0]
            else:
                # print("outm =", out)
                return out
        else:
            # print("out1 =", out)
            return out
    # def __eq__(self, value: object) -> bool:
    #     if isinstance(value, SOp):
    #         if self.primitive != value.primitive:
    #             return False
    #         if self.primitive is jax_pjit.pjit_p:
    #             if self.params["jaxpr"] != value.params["jaxpr"]:
    #                 return False
    #         if len(self.args) != len(value.args):
    #             return False
    #         for (a1, a2) in zip(self.args, value.args):
    #             if a1 != a2:
    #                 return False
    #         return True
    #     return False
    
    # def __hash__(self) -> int:
    #     return hash((primitive_name(self.primitive, self.params), tuple(self.args)))
        
    
# if a functions is jitted then the call to pjit_p.bind is in _python_pjit_helper
# in this helper function the input and output are flattened and unflattend according to in_tree and out_tree
# we do not have access to in_tree and out_tree from the primitive only to in_layout and out_layout
# which specifies the number of input and output arguments of the jaxpr which we call

class SConstant(SExpr):
    def __init__(self, constant) -> None:
        # print("Constant", constant, type(constant), id(constant))
        self.constant = constant
    def __repr__(self) -> str:
        return f"Constant({self.constant})"
    def eval(self, X: dict[str, jax.Array]):
        return self.constant
    # def __eq__(self, value: object) -> bool:
    #     if isinstance(value, SConstant):
    #         if isinstance(self.constant, jax.Array) and isinstance(value.constant, jax.Array):
    #             return bool(jnp.all(self.constant == value.constant))
    #         return self.constant == value.constant
    #     return False
    # def __hash__(self) -> int:
    #     try:
    #         return hash(self.constant)
    #     except TypeError:
    #         return 0
    

class SVar(SExpr):
    def __init__(self, name: str) -> None:
        self.name = name
    def __repr__(self) -> str:
        return self.name
    def eval(self, X: dict[str, jax.Array]) -> jax.Array:
        return X[self.name]
    # def __eq__(self, value: object) -> bool:
    #     if isinstance(value, SVar):
    #         return self.name == value.name
    #     return False
    # def __hash__(self) -> int:
    #     return hash(self.name)

def replace_constant_with_svars(input_object_ids_to_name: Dict[int, str], sexpr: SExpr) -> SExpr:
    if isinstance(sexpr, SConstant):
        # only replace jax.Array
        if isinstance(sexpr.constant, jax.Array) and id(sexpr.constant) in input_object_ids_to_name:
            return SVar(input_object_ids_to_name[id(sexpr.constant)])
        else:
            return sexpr
    elif isinstance(sexpr, SOp):
        sexpr.args = list(map(lambda arg: replace_constant_with_svars(input_object_ids_to_name, arg), sexpr.args))
        return sexpr
    else:
        assert isinstance(sexpr, SVar)
        return sexpr
            

class BranchingDecisions:
    def __init__(self) -> None:
        self.boolean_decisions: List[Tuple[SExpr, bool]] = []
        self.index_decisions: List[Tuple[SExpr, int]] = []


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

    def __bool__(self):
        assert isinstance(self._trace, BranchingTrace)
        boolean_decisions = self._trace.branching_decisions.boolean_decisions
        if self._trace.retrace:
            _, b = boolean_decisions[self._trace.boolean_decision_cnt]
            self._trace.boolean_decision_cnt += 1
            return b
        else:
            concrete_val = jax_core.to_concrete_value(self.val)
            assert concrete_val is not None
            b = bool(concrete_val)
            boolean_decisions.append((self.sexpr, b))
            return b
        
    def __index__(self):
        assert isinstance(self._trace, BranchingTrace)
        index_decisions = self._trace.branching_decisions.index_decisions
        if self._trace.retrace:
            _, b = index_decisions[self._trace.index_decision_cnt]
            self._trace.index_decision_cnt += 1
            return b
        else:
            concrete_val = jax_core.to_concrete_value(self.val)
            assert concrete_val is not None
            b = int(concrete_val)
            index_decisions.append((self.sexpr, b))
            return b

    def full_lower(self):
        return jax_core.full_lower(self.val)
    
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
        self.boolean_decision_cnt = 0
        self.index_decision_cnt = 0

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
