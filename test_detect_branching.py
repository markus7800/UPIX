
import jax
from dccxjax import *
import jax._src.core as jax_core
import jax.numpy as jnp
from typing import List, Tuple, Any


def test_1(x):
    return jax.lax.sin(x)
f = test_1
args = [2.]

@jax.jit
def test_2_inner(d: dict):
    return d["x"] + 2

def test_2(d: dict):
    print("Call test_2 with", d)
    x = test_2_inner(d)
    return jax.lax.sin(x)
f = test_2
args = [{"x": 2.}]

f = test_2
A = jnp.linspace(0.,2*jnp.pi,3)
args = [{"x": A}]

print("Run f:", f(*args))
# exit()

sexprs: List[Tuple[SExpr, Any]] = []
out = detect_branching(f, sexprs)(*args)
print("returns", out)
print("sexprs =", sexprs)
for (sexpr, val) in sexprs:
    print("sexpr =", sexpr)
    print(val, "vs", sexpr.eval({}))
    jaxpr = jax.make_jaxpr(sexpr.eval)({})
    print(jaxpr)
    print("consts =", jaxpr.consts)
    jaxpr_f = jax_core.jaxpr_as_fun(jaxpr)
    print(jaxpr_f())
    print(find_constant(sexpr, A))
    input_object_ids_to_name = {id(A): "A"}
    new_sexpr = replace_constant_with_svars(input_object_ids_to_name, sexpr)
    print("new_sexpr =", new_sexpr)
    new_jaxpr = jax.make_jaxpr(new_sexpr.eval)({"A": A})
    print(new_jaxpr)
    print("consts =", new_jaxpr.consts)
    new_jaxpr_f = jax_core.jaxpr_as_fun(new_jaxpr)
    print(new_jaxpr_f(A + 0.1))

    new_jitted_f = jax.jit(new_sexpr.eval)
    print(new_jitted_f({"A": A}))




