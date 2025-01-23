
class A():
    def __init__(self, val) -> None:
        self.val = val

    def __add__(self, other):
        return A(self.val + other.val)
    
    def __mul__(self, other): ...

    def __getattribute__(self, name):
        print("__getattribute__", name)
        return super().__getattribute__(name)
    
    def __repr__(self) -> str:
        return f"A({self.val})"

setattr(A, "__mul__", lambda x, y: A(x.val * y.val))

print(A(1) + A(2))
print(A(1) * A(2))

print(A.__add__)
print(A.__mul__)
exit()

from dccxjax import *
import jax._src.core as jax_core
import jax.numpy as jnp
import numpy as np

tracer = BranchingTracer(jax_core.eval_trace, jnp.arange(1,3)) # <bound method _forward_operator_to_aval.<locals>.op of Traced<ShapedArray(int32[2])>with<EvalTrace>>
print(tracer.__mul__)
print(tracer.aval._mul)
tracer = BranchingTracer(jax_core.eval_trace, 1) # <bound method _forward_operator_to_aval.<locals>.op of Traced<ShapedArray(int32[], weak_type=True)>with<EvalTrace>>
print(tracer.__mul__)
print(tracer.aval._mul)

# def _forward_operator_to_aval(name):
#   def op(self, *args):
#     return getattr(self.aval, f"_{name}")(self, *args)
#   return op

# --> ufuncs.multiply --> lax.mul --> mul_p.bind