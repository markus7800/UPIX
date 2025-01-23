import jax._src as jax_src
import jax._src.core as jax_core
import jax._src.interpreters.ad as jax_ad
import jax._src.pjit as jax_pjit
from jax._src.dtypes import _jax_types
import jax
from typing import List, Optional, Dict, Tuple, Any, Callable
from abc import ABC, abstractmethod
from jax.tree_util import tree_flatten, tree_unflatten


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
        out = self.primitive.bind(*args, **self.params) # bind insetad of impl is important here to be able to jit-compile it
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
    
# if a functions is jitted then the call to pjit_p.bind is in _python_pjit_helper
# in this helper function the input and output are flattened and unflattend according to in_tree and out_tree
# we do not have access to in_tree and out_tree from the primitive only to in_layout and out_layout
# which specifies the number of input and output arguments of the jaxpr which we call

class SConstant(SExpr):
    def __init__(self, constant) -> None:
        print("Constant", constant, type(constant))
        self.constant = constant
    def __repr__(self) -> str:
        return f"Constant({self.constant})"
    def eval(self, X: dict[str, jax.Array]):
        return self.constant

class SVar(SExpr):
    def __init__(self, name: str) -> None:
        self.name = name
    def __repr__(self) -> str:
        return self.name
    def eval(self, X: dict[str, jax.Array]) -> jax.Array:
        return X[self.name]


def find_constant(sexpr: SExpr, constant):
    if isinstance(sexpr, SConstant):
        if sexpr.constant is constant: # id(sexpr.constant) == id(constant)
            return True, sexpr
    elif isinstance(sexpr, SOp):
        for arg in sexpr.args:
            found, constant = find_constant(arg, constant)
            if found:
                return True, constant
            
    return False, None

def replace_constant_with_svars(input_object_ids_to_name: Dict[int, str], sexpr: SExpr) -> SExpr:
    if isinstance(sexpr, SConstant):
        if id(sexpr.constant) in input_object_ids_to_name:
            return SVar(input_object_ids_to_name[id(sexpr.constant)])
        else:
            return sexpr
    elif isinstance(sexpr, SOp):
        sexpr.args = list(map(lambda arg: replace_constant_with_svars(input_object_ids_to_name, arg), sexpr.args))
        return sexpr
    else:
        assert isinstance(sexpr, SVar)
        return sexpr
            
# based on JVPTrace / JVPTracer

class BranchingTracer(jax_core.Tracer):
    
    def __init__(self, trace: jax_core.Trace, val, sexpr: Optional[SExpr] = None):
        print("Init BranchingTracer with", val, type(val), sexpr)
        self._trace = trace
        self.val = val
        self.sexpr: SExpr = sexpr if sexpr is not None else SConstant(val)

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
# setattr(tracer, f"__{operator_name}__", _forward_operator_to_aval(operator_name)) for operator_name in _array_operators in _set_tracer_aval_forwarding(tracer)


# __getattr__(tracer, name)
#  = getattr(tracer.aval: ShapedArray, name)
#  = setattr(shaped_array, f"_{operator_name}", staticmethod(function)) for operator_name, function in _array_operators.items() in def _set_shaped_array_attributes(shaped_array):
#  = _defer_to_unrecognized_arg(op_symbol, ufunc.op) # Ensure that other array types have the chance to override arithmetic.
#  = @export @partial(jit, inline=True) def op(x: ArrayLike, /) -> Array: ...

def maybe_branching_tracer(trace: jax_core.Trace, val, sexpr: Optional[SExpr] = None):
    try:
        jax_core.get_aval(val)
        return BranchingTracer(trace, val, sexpr)
    except TypeError:
        return val
    
    # try jax_core.get_aval(val)
    if isinstance(val, jax.Array):
        return BranchingTracer(trace, val, sexpr)
    else:
        return val

class BranchingTrace(jax_core.Trace):
    def __init__(self, parent_trace) -> None:
        self.parent_trace = parent_trace

    def process_primitive(self, primitive: jax_core.Primitive, tracers, params):
        print("process_primitive", primitive_name(primitive, params), tracers)
        print(params)
        args = [tracer.val if isinstance(tracer, BranchingTracer) else tracer for tracer in tracers]
        print("args =", args)
        sargs = [tracer.sexpr if isinstance(tracer, BranchingTracer) else SConstant(tracer) for tracer in tracers]
        out = primitive.bind_with_trace(self.parent_trace, args, params)
        sop = SOp(primitive, sargs, params)
        if primitive.multiple_results:
            out_tracer = [maybe_branching_tracer(self, o, sexpr=sop) for o in out]
            print("outm =", out, out_tracer)
        else:
            out_tracer = maybe_branching_tracer(self, out, sexpr=sop)
            print("out1 =", out, out_tracer)
        return out_tracer


def detect_branching(f: Callable, sexprs: List[Tuple[SExpr,Any]]):
    def _f(*args):
        with jax_core.take_current_trace() as parent_trace:
            print("parent_trace:", parent_trace)
            trace = BranchingTrace(parent_trace)
            with jax_core.set_current_trace(trace):
                in_flat, in_tree = tree_flatten(args)
                in_flat = map(lambda x: maybe_branching_tracer(trace, x), in_flat)
                print("in_tree =", in_tree)
                print("in_flat =", in_flat)
                in_tracers = tree_unflatten(in_tree, in_flat)
                out = f(*in_tracers)
                out_flat, out_tree = tree_flatten(out)
                for x in out_flat:
                    if isinstance(x, BranchingTracer):
                        sexprs.append((x.sexpr, x.val))
                out_flat = map(lambda x: x.val if isinstance(x, BranchingTracer) else x, out_flat)
                return tree_unflatten(out_tree, out_flat)
    return _f
