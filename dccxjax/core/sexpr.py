import jax
import jax._src.core as jax_core
import jax._src.pjit as jax_pjit
from abc import ABC, abstractmethod
from typing import List, Dict, Set

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
    @abstractmethod
    def to_human_readable(self) -> str:
        raise NotImplementedError

SUnaryOps = {
}
SBinOps = {
    "less": "<",
    "greater": ">",
    "add": "+",
    "bitwise_and": "&",
}
class SOp(SExpr):
    def __init__(self, primitive: jax_core.Primitive, args: List[SExpr], params) -> None:
        self.primitive = primitive
        self.params = params
        self.args = args
    
    def __repr__(self) -> str:
        return primitive_name(self.primitive, self.params) + "(" + ", ".join(map(repr, self.args)) + ")"
    
    def to_human_readable(self) -> str:
        op_name = self.params["name"] if self.primitive.name == "pjit" else self.primitive.name
        sargs = list(map(lambda a: a.to_human_readable(), self.args))
        if op_name in SUnaryOps:
            return f"{SUnaryOps[op_name]}{sargs[0]}"
        if op_name in SBinOps:
            return f"({sargs[0]} {SBinOps[op_name]} {sargs[1]})"
        return f"{op_name}(" + ", ".join(sargs) + ")"

    def eval(self, X: dict[str, jax.Array]) -> jax.Array:
        args = [arg.eval(X) for arg in self.args]
        # print("eval", primitive_name(self.primitive, self.params), "with", args, self.args, self.params)
        # out = self.primitive.impl(*args, **self.params)
        out = self.primitive.bind(*args, **self.params) # bind instead of impl is important here to be able to jit-compile it later
        # pjit always returns []
        if self.primitive.multiple_results:

            # if a functions is jitted then the call to pjit_p.bind is in _python_pjit_helper
            # in this helper function the input and output are flattened and unflattend according to in_tree and out_tree
            # we do not have access to in_tree and out_tree from the primitive only to in_layout and out_layout
            # which specifies the number of input and output arguments of the jaxpr which we call

            if self.primitive is jax_pjit.pjit_p and self.params["out_layouts"] == (None,):
                # print("out1jit =", out)
                return out[0]
            else:
                # print("outm =", out)
                return out
        else:
            # print("out1 =", out)
            return out
        


class SConstant(SExpr):
    def __init__(self, constant) -> None:
        # print("Constant", constant, type(constant), id(constant))
        self.constant = constant

    def __repr__(self) -> str:
        return f"Constant({self.constant})"
    
    def to_human_readable(self) -> str:
        if isinstance(self.constant, jax.Array):
            if self.constant.shape == ():
                return repr(self.constant.item())
            else:
                return repr(self.constant)
        elif isinstance(self.constant, (bool, int, float)):
            return repr(self.constant)
        else:
            return f"Constant<{type(self.constant)}>"

    def eval(self, X: dict[str, jax.Array]):
        return self.constant
    

class SVar(SExpr):
    def __init__(self, name: str) -> None:
        self.name = name
    def __repr__(self) -> str:
        return self.name
    def to_human_readable(self) -> str:
        return self.name
    def eval(self, X: dict[str, jax.Array]) -> jax.Array:
        return X[self.name]
    

def replace_constants_with_svars(constant_object_ids_to_name: Dict[int, str], sexpr: SExpr, variables: Set[str]) -> SExpr:
    if isinstance(sexpr, SConstant):
        if isinstance(sexpr.constant, jax.Array) and id(sexpr.constant) in constant_object_ids_to_name:
            svar = SVar(constant_object_ids_to_name[id(sexpr.constant)])
            variables.add(svar.name)
            return svar
        return sexpr
    elif isinstance(sexpr, SOp):
        sexpr.args = list(map(lambda arg: replace_constants_with_svars(constant_object_ids_to_name, arg, variables), sexpr.args))
        return sexpr
    else:
        assert isinstance(sexpr, SVar)
        return sexpr