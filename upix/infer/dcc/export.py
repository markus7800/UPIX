import jax
from typing import IO, List, Callable, Tuple, Dict

from jax.tree_util import tree_flatten, tree_unflatten
from jax._src.linear_util import Store

from jax.stages import Traced
from typing import Sequence, Any
from jax._src.export._export import DisabledSafetyCheck, Exported, default_export_platform, check_symbolic_scope_errors, _export_lowered
import jax._src.interpreters.mlir as mlir
import jax._src.config as jax_config

__all__ = [
    
]

def trace_to_flat(f: Callable, args: Tuple) -> Tuple[Traced, Any,Any]:
    args_flat, in_tree = tree_flatten(args)

    tree_store = Store()
    @jax.jit
    def f_flat(*_args_flat):
        _args = tree_unflatten(in_tree, _args_flat)
        out = f(*_args)
        out_flat, out_tree = tree_flatten(out)
        tree_store.store((in_tree, out_tree))
        return out_flat
    
    traced = f_flat.trace(*args_flat)
    # print(traced.jaxpr)
    assert isinstance(tree_store.val, Tuple)

    check_symbolic_scope_errors(f_flat, args_flat, {})

    return traced, *tree_store.val

# fun does not have to be jit-wrapped
def export_flat(
    fun: Callable,
    platforms: Sequence[str] | None = None,
    disabled_checks: Sequence[DisabledSafetyCheck] = (),
    _device_assignment_for_internal_jax2tf_use_only=None,
    override_lowering_rules=None,
    ) -> Callable[..., Tuple[Exported,Any,Any]]:
  
  
  def do_export(*args_specs) -> Tuple[Exported, Any,Any]:
    if platforms is not None:
      actual_lowering_platforms = tuple(platforms)
    else:
      actual_lowering_platforms = (default_export_platform(),)

    # check_symbolic_scope_errors(fun_jit, args_specs, kwargs_specs)

    traced, in_tree, out_tree = trace_to_flat(fun, args_specs)

    lowered = traced.lower(
        lowering_platforms=actual_lowering_platforms,
        _private_parameters=mlir.LoweringParameters(
            override_lowering_rules=override_lowering_rules,
            for_export=True,
            export_ignore_forward_compatibility=jax_config.export_ignore_forward_compatibility.value))
    export_lowered = _export_lowered(
        lowered, traced.jaxpr, traced.fun_name,
        disabled_checks=disabled_checks,
        _device_assignment_for_internal_jax2tf_use_only=_device_assignment_for_internal_jax2tf_use_only)
    return export_lowered, in_tree, out_tree
  return do_export