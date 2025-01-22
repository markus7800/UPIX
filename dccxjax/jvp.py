from jax._src.dtypes import dtype, float0

import jax._src.core as jax_core
import jax._src.interpreters.ad as jax_ad
import jax._src.ad_util as jax_ad_util
import jax._src.tree_util as jax_tree_util
import jax._src.api_util as jax_api_util
import jax._src.api as jax_api
from jax._src import linear_util as lu
from jax._src.lax import lax as lax_internal
from functools import partial, lru_cache
import jax

from typing import TypeVar, Iterable, Callable, Sequence

class JVPTracer(jax_core.Tracer):
  __slots__ = ['primal', 'tangent']

  def __init__(self, trace, primal, tangent):
    # print("Init JVPTracer", primal)
    self._trace = trace
    self.primal = primal
    self.tangent = tangent

  @property
  def aval(self):
    return jax_core.get_aval(self.primal)

  def full_lower(self):
    if type(self.tangent) is jax_ad_util.Zero:
      return jax_core.full_lower(self.primal)
    else:
      return self

  def to_concrete_value(self):
    return jax_core.to_concrete_value(self.primal)
  
T1 = TypeVar("T1")
T2 = TypeVar("T2")
def unzip2(xys: Iterable[tuple[T1, T2]]
    ) -> tuple[tuple[T1, ...], tuple[T2, ...]]:
  """Unzip sequence of length-2 tuples into two tuples."""
  # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
  # is too permissive about inputs, and does not guarantee a length-2 output.
  xs: list[T1] = []
  ys: list[T2] = []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return tuple(xs), tuple(ys)

class JVPTrace(jax_core.Trace):
  def __init__(self, parent_trace, tag):
    self.tag = tag
    self.parent_trace = parent_trace

  def to_primal_tangent_pair(self, val):
    if isinstance(val, JVPTracer) and isinstance(val._trace, JVPTrace) and val._trace.tag is self.tag:
      return (val.primal, val.tangent)
    else:
      tangent_zero = jax_ad_util.Zero.from_primal_value(val)
      return (val, tangent_zero)

  def process_primitive(self, primitive, tracers, params):
    primals_in, tangents_in = unzip2(map(self.to_primal_tangent_pair, tracers))
    # print(f"JVPTrace process_primitive {primitive} tracers_in =", tracers)
    if all(type(t) is jax_ad_util.Zero for t in tangents_in):
      return primitive.bind_with_trace(self.parent_trace, primals_in, params)
    jvp = jax_ad.primitive_jvps.get(primitive)
    if not jvp:
      msg = f"Differentiation rule for '{primitive}' not implemented"
      raise NotImplementedError(msg)
    with jax_core.set_current_trace(self.parent_trace):
      primal_out, tangent_out = jvp(primals_in, tangents_in, **params)

    if primitive.multiple_results:
      out = [maybe_jvp_tracer(self, x, t) for x, t in zip(primal_out, tangent_out)]
    else:
      out = maybe_jvp_tracer(self, primal_out, tangent_out)
    
    # print(f"JVPTrace process_primitive {primitive} tracers_out =", out)
    return out
  
def maybe_jvp_tracer(trace, primal, tangent):
  if type(tangent) is jax_ad_util.Zero or dtype(tangent) == float0:
    return primal
  else:
    return JVPTracer(trace, primal, tangent)
  

def jvp_v1(f, primals, tangents):
  tag = jax_core.TraceTag()
  with jax_core.take_current_trace() as parent_trace:
    trace = JVPTrace(parent_trace, tag)
    # print("jvp_subtrace", primals)
    in_tracers = [maybe_jvp_tracer(trace, x, t)
                  for x, t in zip(primals, tangents)]
    # print("jvp_subtrace", in_tracers)
    # print("jvp_subtrace primal", [t.primal for t in in_tracers])
    # print("jvp_subtrace aval", [t.aval for t in in_tracers])
    with jax_core.set_current_trace(trace):
      ans = f(*in_tracers)
      return ans
    out = unzip2(map(trace.to_primal_tangent_pair, ans))
  return out
  

def mygrad(f: Callable):
    def fun_grad(*args, **kwargs):
        fun = lu.wrap_init(f, params=kwargs)
        primals_flat, in_tree = jax_tree_util.tree_flatten(*args)
        flat_fun, out_tree = jax_api_util.flatten_fun_nokwargs(fun, in_tree)
        return jvp_v1(flat_fun, primals_flat, lax_internal._one(primals_flat))
    return fun_grad

def _vjp(fun: lu.WrappedFun, *primals):
    primals_flat, in_tree = jax_tree_util.tree_flatten(primals)
    flat_fun, out_tree = jax_api_util.flatten_fun_nokwargs(fun, in_tree)
    out_primals, vjp = jax_ad.vjp(flat_fun, primals_flat)
    out_tree = out_tree()

    out_primal_avals = map(jax_core.shaped_abstractify, out_primals)
    out_primal_py = jax_tree_util.tree_unflatten(out_tree, out_primals)
    # vjp_py = jax_tree_util.Partial(partial(jax_api._vjp_pullback_wrapper, fun.__name__,
    #                         out_primal_avals, (out_tree, in_tree)), vjp)
    vjp_py = vjp
    
    return out_primal_py, vjp_py


def grad(fun: Callable, argnums: int | Sequence[int] = 0):
  def f_grad(*args, **kwargs):
    f = lu.wrap_init(fun, params=kwargs)
    f_partial, dyn_args = jax_api_util.argnums_partial(f, argnums, args, require_static_args_hashable=False)
    ans, vjp_py = _vjp(f_partial, *dyn_args)
    print(lax_internal._one(ans))
    g = vjp_py(lax_internal._one(ans))
    return g
  return f_grad